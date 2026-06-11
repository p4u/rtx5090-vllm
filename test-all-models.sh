#!/usr/bin/env bash
# Boot each configured model, probe health, run a one-shot completion, stop.
# Results are printed as a table at the end.
#
# Usage:
#   ./test-all-models.sh               # test all models
#   ./test-all-models.sh <model> ...   # test specific models
#
# Env overrides:
#   ENDPOINT   base URL of the server  (default: http://localhost:8080/v1)
#   TIMEOUT    seconds to wait for ready (default: 180)
#   MAX_TOKENS tokens to generate in the test completion (default: 2000)

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Bind to localhost so the health check works from this host.
export HOST_IP="${HOST_IP:-127.0.0.1}"
export HOST_PORT="${HOST_PORT:-8080}"
ENDPOINT="${ENDPOINT:-http://${HOST_IP}:${HOST_PORT}/v1}"
TIMEOUT="${TIMEOUT:-180}"
# 2000 tokens: reasoning MoEs can burn 1000-1500 tokens thinking before content.
MAX_TOKENS="${MAX_TOKENS:-2000}"
PROMPT="Reply with exactly one sentence confirming you are working."

ALL_MODELS=(
  qwen36-27b-awq
  qwen36-27b-nvfp4
  cascade2
  qwen36
  qwen3-coder
  gemma4
  gemma4-text
  gpt-oss
  nemotron3
)

MODELS=("${@:-${ALL_MODELS[@]}}")

# --- helpers -----------------------------------------------------------------

wait_ready() {
  local deadline=$(( $(date +%s) + TIMEOUT ))
  while true; do
    if curl -sf --max-time 5 "$ENDPOINT/models" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) > deadline )); then
      return 1
    fi
    sleep 3
  done
}

run_completion() {
  local resp
  resp=$(curl -sS --max-time 60 "$ENDPOINT/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
          --arg p "$PROMPT" \
          --argjson mt "$MAX_TOKENS" \
          '{model:"default", max_tokens:$mt, temperature:0.1,
            messages:[{role:"user",content:$p}]}')")
  local content
  content=$(printf '%s' "$resp" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
  if [[ -z "$content" ]]; then
    local err
    err=$(printf '%s' "$resp" | jq -r '.error.message // empty' 2>/dev/null || true)
    if [[ -n "$err" ]]; then
      echo "ERROR: $err"
    else
      echo "ERROR: no content (finish=$(printf '%s' "$resp" | jq -r '.choices[0].finish_reason // "?" ' 2>/dev/null), raw=$(printf '%s' "$resp" | jq -c '.' 2>/dev/null | cut -c1-200))"
    fi
    return 1
  fi
  printf '%s' "$content" | tr '\n' ' ' | cut -c1-120
  return 0
}

stop_server() {
  bash ./stop-vllm.sh >/dev/null 2>&1 || true
  local deadline=$(( $(date +%s) + 15 ))
  while ss -tlnp 2>/dev/null | grep -q ":${HOST_PORT} "; do
    if (( $(date +%s) > deadline )); then break; fi
    sleep 1
  done
}

# --- main loop ---------------------------------------------------------------

declare -A RESULTS
declare -A OUTPUTS

echo "=========================================================="
echo " vLLM model test — $(date '+%Y-%m-%d %H:%M')"
echo " image : $(docker images vllm/vllm-openai:latest --format '{{.CreatedSince}} ({{.ID}})')"
echo " models: ${MODELS[*]}"
echo "=========================================================="

for model in "${MODELS[@]}"; do
  echo
  echo "── $model ──────────────────────────────────────────────"

  echo "  starting..."
  start_out=$(bash ./run.sh "$model" -d 2>&1) || true
  if echo "$start_out" | grep -qi "error\|failed"; then
    echo "$start_out" | tail -5 | sed 's/^/  /'
    RESULTS[$model]="FAIL"
    OUTPUTS[$model]="failed to start container"
    stop_server
    continue
  fi

  echo "  waiting for /v1/models (up to ${TIMEOUT}s)..."
  if ! wait_ready; then
    echo "  TIMEOUT — server did not become ready"
    RESULTS[$model]="FAIL"
    OUTPUTS[$model]="timeout waiting for server"
    echo "  --- last container logs ---"
    docker logs --tail 20 vllm 2>&1 | sed 's/^/  /' || true
    stop_server
    continue
  fi
  echo "  server ready"

  echo "  running completion..."
  output=""
  if output=$(run_completion); then
    echo "  PASS: $output"
    RESULTS[$model]="PASS"
    OUTPUTS[$model]="$output"
  else
    echo "  FAIL: $output"
    RESULTS[$model]="FAIL"
    OUTPUTS[$model]="$output"
  fi

  stop_server
done

# --- summary -----------------------------------------------------------------

echo
echo "=========================================================="
echo " SUMMARY"
echo "=========================================================="
pass=0; fail=0
for model in "${MODELS[@]}"; do
  status="${RESULTS[$model]:-SKIP}"
  output="${OUTPUTS[$model]:-}"
  printf "  %-22s  %s\n" "$model" "$status"
  if [[ "$status" == "FAIL" ]]; then
    printf "                           → %s\n" "$output"
    (( fail++ )) || true
  else
    (( pass++ )) || true
  fi
done
echo "----------------------------------------------------------"
echo "  passed: $pass / $(( pass + fail ))"
echo "=========================================================="
