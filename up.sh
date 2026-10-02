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
#   ./up.sh                  # default model (see DEFAULT_MODEL)
#   MODEL=qwen38-27b ./up.sh # any key from ./run.sh --list
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Bonsai 2: smallest download in the lineup (7.4 GB) and the fastest decode, so
# it is the least painful first run. Override with MODEL=<key>.
DEFAULT_MODEL="bonsai2"
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

# Where the UI answers on THIS host. TLS mode always serves 443 (see run-ui.sh);
# probe loopback either way so DNS and public routing can't affect readiness.
if [[ "${UI_TLS:-}" == "1" || "${UI_TLS:-}" == "letsencrypt" ]] && [[ -n "${UI_DOMAIN:-}" ]]; then
  UI_LOCAL="https://127.0.0.1"
  UI_PUBLIC="https://${UI_DOMAIN}"
else
  UI_LOCAL="http://127.0.0.1:${UI_PORT:-8090}"
  _disp="${UI_HOST:-0.0.0.0}"; [[ "$_disp" == "0.0.0.0" ]] && _disp="localhost"
  UI_PUBLIC="http://${_disp}:${UI_PORT:-8090}"
fi

ui_answers() { curl -skf --max-time 4 "$UI_LOCAL/" >/dev/null 2>&1; }

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

# The UI mints the bearer token that gates its OpenAI proxy; handing the same
# token to the model keeps the model's own port from being open tokenless,
# exactly as a launch from the UI would. state.json exists once the app has
# imported, which the readiness probe above has already proven.
TOKEN="$(python3 -c "import json;print(json.load(open('ui/data/state.json'))['api_token'])" 2>/dev/null || true)"
[[ -n "$TOKEN" ]] || echo "up.sh: warning — could not read the UI token; the model port will not be gated" >&2

# ── 3. the model ────────────────────────────────────────────────────────────
running_key="$(docker inspect -f '{{index .Config.Labels "vllm.model-key"}}' vllm 2>/dev/null || true)"
running_state="$(docker inspect -f '{{.State.Running}}' vllm 2>/dev/null || true)"
if [[ "$running_key" == "$MODEL" && "$running_state" == "true" ]]; then
  step "Model '$MODEL' already running — leaving it alone"
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
