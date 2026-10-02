#!/usr/bin/env bash
# `make` / `make up` — take this host from nothing to a served model behind the
# web UI, doing every intermediate step.
#
#   1. preflight.sh      — halt early if something required is missing
#   2. run-ui.sh         — build + start the UI, obscura and the Python sandbox
#   3. run.sh <model>    — build the runtime image and download weights if
#                          missing, then serve (gated by the UI's own token, so
#                          the model port is never open without one)
#   4. wait for the container healthcheck and print where everything is
#
# Already-running pieces are left alone, so this is safe to re-run: it only
# does the work that is actually missing.
#
# Usage:
#   ./up.sh                  # keep the current model; DEFAULT_MODEL on a fresh host
#   MODEL=qwen38-27b ./up.sh # serve this key (replaces the current model)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Bonsai 2: smallest download in the lineup (7.4 GB) and the fastest decode, so
# it is the least painful first run. Used only when no model container exists
# yet; MODEL=<key> always wins (and replaces whatever is serving).
DEFAULT_MODEL="bonsai2"
MODEL_EXPLICIT="${MODEL:+1}"
MODEL="${MODEL:-$DEFAULT_MODEL}"
WAIT_HEALTHY_S="${WAIT_HEALTHY_S:-900}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# ── 1. can this host do it at all? ──────────────────────────────────────────
step "Checking prerequisites"
./preflight.sh

# Same .env loader as run.sh/run-ui.sh: KEY=value lines, real environment wins.
if [[ -f .env ]]; then
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
  done < .env
  unset _line _key _val
fi

# Where the UI answers on THIS host. It binds UI_HOST only (ui/serve.py), so a
# UI_HOST of a VPN address is NOT reachable on loopback — probe UI_HOST itself
# unless it is the wildcard. Try https:443 and plain http:UI_PORT both: with
# UI_TLS set, run-ui.sh still falls back to plain http when no certificate can
# be obtained, and readiness must follow what actually came up, not the config.
ui_addr="${UI_HOST:-0.0.0.0}"
[[ "$ui_addr" == "0.0.0.0" || "$ui_addr" == "::" ]] && ui_addr="127.0.0.1"
[[ "$ui_addr" == *:* ]] && ui_addr="[$ui_addr]"        # IPv6 literal in a URL
UI_CANDIDATES=("http://${ui_addr}:${UI_PORT:-8090}")
if [[ "${UI_TLS:-}" == "1" || "${UI_TLS:-}" == "letsencrypt" ]] && [[ -n "${UI_DOMAIN:-}" ]]; then
  UI_CANDIDATES=("https://${ui_addr}" "${UI_CANDIDATES[@]}")
fi
UI_LOCAL=""
ui_answers() {
  local u
  for u in "${UI_CANDIDATES[@]}"; do
    curl -skf --max-time 4 "$u/" >/dev/null 2>&1 && { UI_LOCAL="$u"; return 0; }
  done
  return 1
}

# ── 2. the UI ───────────────────────────────────────────────────────────────
if [[ "$(docker inspect -f '{{.State.Running}}' vllm-ui 2>/dev/null)" == "true" ]] && ui_answers; then
  step "Web UI already running — leaving it alone"
else
  step "Starting the web UI (first run also builds the UI + sandbox images)"
  ./run-ui.sh
  printf '    waiting for the UI to answer'
  for _ in $(seq 1 60); do ui_answers && break; printf '.'; sleep 2; done
  printf '\n'
  ui_answers || { echo "up.sh: the UI did not come up — check ./logs-ui.sh" >&2; exit 1; }
fi

# What to tell the user follows what answered: https means the certificate
# worked, so the public name applies; plain http reports the bind address.
if [[ "$UI_LOCAL" == https://* ]]; then
  UI_PUBLIC="https://${UI_DOMAIN}"
else
  _disp="${UI_HOST:-0.0.0.0}"; [[ "$_disp" == "0.0.0.0" ]] && _disp="localhost"
  UI_PUBLIC="http://${_disp}:${UI_PORT:-8090}"
  [[ "${UI_TLS:-}" == "1" || "${UI_TLS:-}" == "letsencrypt" ]] \
    && echo "up.sh: warning — UI_TLS is set but the UI is serving plain http (no certificate; see make ui-logs)" >&2
fi

# The UI mints the bearer token that gates its OpenAI proxy; handing the same
# token to the model keeps the model's own port from being open tokenless,
# exactly as a launch from the UI would. state.json exists once the app has
# imported, which the readiness probe above has already proven.
TOKEN="$(python3 -c "import json;print(json.load(open('ui/data/state.json'))['api_token'])" 2>/dev/null || true)"
[[ -n "$TOKEN" ]] || echo "up.sh: warning — could not read the UI token; the model port will not be gated" >&2

# ── 3. the model ────────────────────────────────────────────────────────────
# Only an EXPLICIT MODEL= switches what is served. A bare `make` re-run must not
# swap out a model someone picked in the UI for the default, so without one it
# keeps whatever the container already holds (restarting it if it is stopped),
# and falls back to DEFAULT_MODEL only when there is no model container at all.
running_key="$(docker inspect -f '{{index .Config.Labels "vllm.model-key"}}' vllm 2>/dev/null || true)"
running_state="$(docker inspect -f '{{.State.Running}}' vllm 2>/dev/null || true)"
[[ "$running_key" == "<no value>" ]] && running_key=""
if [[ -z "$MODEL_EXPLICIT" && -n "$running_key" ]]; then
  MODEL="$running_key"
fi
if [[ "$running_key" == "$MODEL" && "$running_state" == "true" ]]; then
  step "Model '$MODEL' already running — leaving it alone"
  [[ -z "$MODEL_EXPLICIT" ]] && echo "    (switch with: make up MODEL=<key>; keys: make list)"
else
  step "Starting model '$MODEL'"
  echo "    first run for a model downloads its weights, and a llama.cpp-backed"
  echo "    one also compiles its image (~20 min) — later runs skip both."
  VLLM_API_KEY="${TOKEN:-${VLLM_API_KEY:-}}" ./run.sh "$MODEL"
fi

# ── 4. wait for it to actually serve ────────────────────────────────────────
step "Waiting for '$MODEL' to report healthy"
deadline=$(( $(date +%s) + WAIT_HEALTHY_S ))
last=""
while :; do
  health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' vllm 2>/dev/null || echo missing)"
  state="$(docker inspect -f '{{.State.Status}}' vllm 2>/dev/null || echo missing)"
  if [[ "$health" == "healthy" ]]; then printf '\n'; break; fi
  if [[ "$state" == "exited" || "$state" == "missing" ]]; then
    printf '\n'
    echo "up.sh: the model container is $state — last log lines:" >&2
    docker logs --tail 20 vllm 2>&1 | sed 's/^/    /' >&2
    exit 1
  fi
  if (( $(date +%s) > deadline )); then
    printf '\n'
    echo "up.sh: '$MODEL' was still '$health' after ${WAIT_HEALTHY_S}s — watch it with: make logs" >&2
    exit 1
  fi
  [[ "$health" != "$last" ]] && { printf '    %s' "$health"; last="$health"; } || printf '.'
  sleep 5
done

# ── done ────────────────────────────────────────────────────────────────────
printf '\n\033[1m==> Up.\033[0m\n\n'
echo "  web UI      : $UI_PUBLIC/"
echo "  OpenAI API  : $UI_PUBLIC/v1   (bearer token — API access panel in the UI)"
echo "  model       : $MODEL"
echo ""
echo "  log in with UI_PASSWORD, then use the Chat tab or point any OpenAI"
echo "  client at the URL above."
echo ""
echo "  make status | make logs | make ui-logs | make stop"
