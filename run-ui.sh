#!/usr/bin/env bash
# Launch the vllm-ui container: web UI for switching/tuning models, a live
# dashboard (vLLM /metrics + GPU), and the token-gated OpenAI proxy on
# ${UI_HOST:-0.0.0.0}:${UI_PORT:-8090}.
#
# How it can manage models from inside a container:
#   • /var/run/docker.sock is mounted → it drives the HOST's docker daemon.
#   • The repo is mounted at its IDENTICAL host path and used as workdir →
#     when the UI runs ./run.sh, the `-v $SCRIPT_DIR/cache:…` binds run.sh
#     issues are host-valid paths, so the daemon resolves them correctly.
#   • --network host → the UI reaches vLLM on host loopback. Models launched
#     from the UI get HOST_IP=127.0.0.1 forced, so raw vLLM is loopback-only
#     and the token-gated proxy (/v1/* with Authorization: Bearer) is the only
#     LAN entrance. Manual ./run.sh from a shell is unchanged.
#   • --user $(id -u) + the docker group → files created via the mount
#     (ui/data/, downloaded weights, logs/) stay owned by you, not root.
#
# Config (via .env or environment; see .env.example):
#   UI_PASSWORD   REQUIRED — gates the web UI.
#   UI_HOST       bind address (default 0.0.0.0; host networking, so this is
#                 the real bind — use a VPN address to restrict like BIND_CIDR).
#   UI_PORT       port for plain-http mode (default 8090; ignored with TLS).
#   UI_TLS=1      Let's Encrypt TLS — serves https://UI_DOMAIN/ on 443.
#
# Without TLS, password and token travel in plaintext — keep it on a VPN.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Same .env loader as run.sh: KEY=value lines, real environment wins.
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    _line="${_line%$'\r'}"
    [[ "$_line" =~ ^[[:space:]]*(#|$) ]] && continue
    _line="${_line#export }"
    [[ "$_line" == *=* ]] || continue
    _key="${_line%%=*}"; _key="${_key//[[:space:]]/}"
    [[ "$_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!_key+x}" ]] && continue
    _val="${_line#*=}"
    _val="${_val%\"}"; _val="${_val#\"}"
    _val="${_val%\'}"; _val="${_val#\'}"
    export "$_key=$_val"
  done < "$SCRIPT_DIR/.env"
  unset _line _key _val
fi

if [[ -z "${UI_PASSWORD:-}" ]]; then
  echo "run-ui.sh: UI_PASSWORD is not set — refusing to start an unprotected UI." >&2
  echo "  Set it in .env (see .env.example) or: UI_PASSWORD=... ./run-ui.sh" >&2
  exit 1
fi

if [[ ! -S /var/run/docker.sock ]]; then
  echo "run-ui.sh: /var/run/docker.sock not found — is Docker running?" >&2
  exit 1
fi
DOCKER_GID="$(stat -c %g /var/run/docker.sock)"

# ui/data must exist BEFORE the mount so it's created with your uid, not root.
mkdir -p "$SCRIPT_DIR/ui/data"

# ─── TLS (UI_TLS=1 + UI_DOMAIN) — https on 443, and ONLY on 443 ─────────────
# Real Let's Encrypt certificate via acme.sh TLS-ALPN-01: the CA connects to
# PUBLIC port 443 of UI_DOMAIN — open it in your firewall. With TLS active
# the UI itself serves https on 443 (UI_PORT is ignored; URLs need no port
# suffix) and plain http hitting 443 gets redirected. Renewal is automatic:
# a daily task inside the UI launches a detached helper that briefly stops
# the UI (frees :443 for the ALPN responder), renews, and starts it again —
# ~30s of downtime every ~60 days. acme.sh state: ui/data/acme-sh
# (gitignored). A best-effort redirector on :80 also sends http → https.
TLS_ACTIVE=""
CERT_DIR="$SCRIPT_DIR/ui/data/certs"
ACMESH_DIR="$SCRIPT_DIR/ui/data/acme-sh"
if [[ "${UI_TLS:-}" == "1" || "${UI_TLS:-}" == "letsencrypt" ]]; then
  if [[ -z "${UI_DOMAIN:-}" || "${UI_DOMAIN}" == http* ]]; then
    echo "run-ui.sh: UI_TLS needs UI_DOMAIN set to a bare hostname — starting without TLS" >&2
  else
    LIVE="$CERT_DIR/live/$UI_DOMAIN"
    mkdir -p "$LIVE" "$ACMESH_DIR"
    if [[ ! -f "$LIVE/fullchain.pem" ]]; then
      # a running UI occupies :443 — free it for the ALPN responder
      docker rm -f vllm-ui vllm-ui-redirect >/dev/null 2>&1 || true
      echo ">>> requesting Let's Encrypt certificate for $UI_DOMAIN (TLS-ALPN-01 on public port 443)..."
      docker run --rm --network host -v "$ACMESH_DIR:/acme.sh" neilpang/acme.sh \
        --issue --alpn -d "$UI_DOMAIN" --server letsencrypt \
        ${TLS_EMAIL:+-m "$TLS_EMAIL"} || true
      # acme.sh issues ECC certs by default → --ecc when exporting
      docker run --rm -v "$ACMESH_DIR:/acme.sh" -v "$CERT_DIR:/certs" neilpang/acme.sh \
        --install-cert -d "$UI_DOMAIN" --ecc \
        --fullchain-file "/certs/live/$UI_DOMAIN/fullchain.pem" \
        --key-file "/certs/live/$UI_DOMAIN/privkey.pem" >/dev/null 2>&1 || true
    fi
    if [[ -f "$LIVE/fullchain.pem" ]]; then
      # acme.sh containers write as root; the UI runs as you
      docker run --rm -v "$CERT_DIR:/c" -v "$ACMESH_DIR:/a" alpine \
        chown -R "$(id -u):$(id -g)" /c /a
      TLS_ACTIVE=1
      # the renewal helper needs this image at 3am, not at first failure
      docker pull -q docker:cli >/dev/null 2>&1 || true
      [[ -n "${UI_PORT:-}" && "${UI_PORT}" != "443" ]] \
        && echo ">>> note: UI_PORT=$UI_PORT is ignored — TLS always serves on 443" >&2
      echo ">>> TLS active: https://$UI_DOMAIN/ (port 443)"
    else
      echo ">>> warning: no certificate obtained (is public port 443 reachable for $UI_DOMAIN?) — starting WITHOUT TLS" >&2
    fi
  fi
elif [[ -n "${UI_TLS:-}" && "${UI_TLS}" != "0" ]]; then
  echo "run-ui.sh: unknown UI_TLS value '${UI_TLS}' (use 1) — starting without TLS" >&2
fi

# http:80 → https redirector (root: port 80 is privileged; fixed inline
# script, no repo access). Only when TLS is active.
docker rm -f vllm-ui-redirect >/dev/null 2>&1 || true
if [[ -n "$TLS_ACTIVE" ]]; then
  docker run -d --name vllm-ui-redirect --network host --restart unless-stopped \
    -e "UI_DOMAIN=$UI_DOMAIN" -e "UI_PORT=${UI_PORT:-8090}" \
    python:3.12-slim python3 -c '
import http.server, os
DOMAIN, PORT = os.environ["UI_DOMAIN"], os.environ["UI_PORT"]
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(301)
        self.send_header("Location", f"https://{DOMAIN}{self.path}")
        self.end_headers()
    do_HEAD = do_GET
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("0.0.0.0", 80), H).serve_forever()' >/dev/null \
    || echo ">>> note: :80 redirector not started (port busy?)" >&2
fi

echo ">>> building vllm-ui image..."
docker build -q -t vllm-ui "$SCRIPT_DIR/ui" >/dev/null

# obscura powers the chat's web-browsing tools (one `docker run --rm` per
# search/fetch). Missing image = browsing shows as unavailable, UI still runs.
docker pull -q h4ckf0r0day/obscura >/dev/null 2>&1 \
  || echo ">>> warning: could not pull h4ckf0r0day/obscura — chat web browsing will be unavailable" >&2

docker rm -f vllm-ui >/dev/null 2>&1 || true

docker run -d \
  --name vllm-ui \
  --network host \
  --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  --group-add "$DOCKER_GID" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$SCRIPT_DIR:$SCRIPT_DIR" \
  -w "$SCRIPT_DIR" \
  -e UI_PASSWORD \
  -e UI_HOST \
  -e UI_PORT \
  -e UI_DOMAIN \
  -e UI_TLS \
  -e "UI_TLS_ACTIVE=$TLS_ACTIVE" \
  vllm-ui >/dev/null

display_host="${UI_HOST:-0.0.0.0}"
[[ "$display_host" == "0.0.0.0" ]] && display_host="localhost"
if [[ -n "$TLS_ACTIVE" ]]; then
  echo ">>> vllm-ui started"
  echo ">>> web UI      : https://$UI_DOMAIN/"
  echo ">>> OpenAI API  : https://$UI_DOMAIN/v1  (Bearer token — see UI)"
else
  echo ">>> vllm-ui started"
  echo ">>> web UI      : http://${display_host}:${UI_PORT:-8090}/"
  echo ">>> OpenAI API  : http://${display_host}:${UI_PORT:-8090}/v1  (Bearer token — see UI)"
fi
echo ">>> logs / stop : ./logs-ui.sh | ./stop-ui.sh"
