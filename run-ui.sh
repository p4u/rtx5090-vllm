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
#   UI_PORT       port (default 8090).
#
# SECURITY: no TLS — password and token travel in plaintext. Front it with a
# VPN (WireGuard) or a TLS reverse proxy before exposing beyond a LAN.
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

echo ">>> building vllm-ui image..."
docker build -q -t vllm-ui "$SCRIPT_DIR/ui" >/dev/null

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
  vllm-ui >/dev/null

display_host="${UI_HOST:-0.0.0.0}"
[[ "$display_host" == "0.0.0.0" ]] && display_host="localhost"
echo ">>> vllm-ui started"
echo ">>> web UI      : http://${display_host}:${UI_PORT:-8090}/"
echo ">>> OpenAI API  : http://${display_host}:${UI_PORT:-8090}/v1  (Bearer token — see UI)"
echo ">>> logs / stop : ./logs-ui.sh | ./stop-ui.sh"
