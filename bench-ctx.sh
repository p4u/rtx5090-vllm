#!/usr/bin/env bash
# One-shot context-ceiling benchmark for all vLLM models.
# Boots each model at its configured context, records KV pool size, runs a
# completion, then moves to the next model.
#
# Usage:
#   ./bench-ctx.sh               # all models
#   ./bench-ctx.sh gemma4 qwen36 # specific models
#
# Output: bench-ctx-results.txt

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export HOST_IP="${HOST_IP:-127.0.0.1}"
export HOST_PORT="${HOST_PORT:-8080}"
ENDPOINT="http://${HOST_IP}:${HOST_PORT}/v1"
TIMEOUT=300
RESULTS_FILE="$SCRIPT_DIR/bench-ctx-results.txt"

PROMPT="Reply with exactly one sentence confirming you are working."
MAX_TOKENS=2000  # enough for reasoning models

# Format: "key|target_ctx|notes" — target_ctx matches run.sh's configured ceiling.
ALL_MODELS=(
  "qwen36-27b-awq|262144|DeltaNet 27B AWQ"
  "qwen36-27b-nvfp4|262144|DeltaNet 27B NVFP4"
  "qwen36|196608|DeltaNet 35B MoE (tight VRAM)"
  "qwen3-coder|221184|30B Coder MoE"
  "cascade2|131072|Mamba2+MoE 30B"
  "gemma4|262144|Gemma4 MoE 26B"
  "gemma4-text|32768|Gemma4 dense 27B (KV limited)"
  "gpt-oss|131072|gpt-oss-20b (hard YaRN limit)"
  "nemotron3|229376|Nemotron3 Omni 31B (mm disabled)"
)

if [[ $# -gt 0 ]]; then
  MODELS=()
  for arg in "$@"; do
    for entry in "${ALL_MODELS[@]}"; do
      if [[ "${entry%%|*}" == "$arg" ]]; then
        MODELS+=("$entry")
      fi
    done
  done
else
  MODELS=("${ALL_MODELS[@]}")
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

wait_ready() {
  local deadline=$(( $(date +%s) + TIMEOUT ))
  while true; do
    if curl -sf --max-time 5 "$ENDPOINT/models" >/dev/null 2>&1; then return 0; fi
    if (( $(date +%s) > deadline )); then return 1; fi
    sleep 3
  done
}

run_completion() {
  local resp
  resp=$(curl -sS --max-time 120 "$ENDPOINT/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$PROMPT" --argjson mt "$MAX_TOKENS" \
          '{model:"default",max_tokens:$mt,temperature:0.1,
            messages:[{role:"user",content:$p}]}')")
  local content
  content=$(printf '%s' "$resp" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
  if [[ -z "$content" ]]; then
    local err
    err=$(printf '%s' "$resp" | jq -r '.error.message // empty' 2>/dev/null || true)
    echo "FAIL: ${err:-no content ($(printf '%s' "$resp" | jq -c '.' 2>/dev/null | cut -c1-150))}"
    return 1
  fi
  printf '%s' "$content" | tr '\n' ' ' | cut -c1-100
}

stop_server() {
  bash ./stop-vllm.sh >/dev/null 2>&1 || true
  local deadline=$(( $(date +%s) + 20 ))
  while ss -tlnp 2>/dev/null | grep -q ":${HOST_PORT} "; do
    if (( $(date +%s) > deadline )); then break; fi
    sleep 1
  done
}

get_kv_info() {
  docker logs vllm 2>&1 | grep -E "KV cache size|Available KV cache|GPU KV cache" | tail -3 | sed 's/^.*INFO[^]]*\] //'
}

# ── Main loop ─────────────────────────────────────────────────────────────────

{
  echo "============================================================"
  echo " vLLM context ceiling benchmark — $(date '+%Y-%m-%d %H:%M')"
  echo " image: $(docker images vllm/vllm-openai:latest --format '{{.CreatedSince}} ({{.ID}})')"
  echo "============================================================"
} | tee "$RESULTS_FILE"

declare -A RESULTS
declare -A KV_POOLS
declare -A CTX_USED

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  rest="${entry#*|}"
  target_ctx="${rest%%|*}"
  notes="${rest#*|}"

  echo "" | tee -a "$RESULTS_FILE"
  echo "── $model ($notes) ──────────────────────────────────" | tee -a "$RESULTS_FILE"
  echo "   target ctx: $target_ctx" | tee -a "$RESULTS_FILE"

  stop_server

  echo "   starting..." | tee -a "$RESULTS_FILE"
  start_out=$(bash ./run.sh "$model" -d 2>&1) || true
  if echo "$start_out" | grep -qi "error\|failed"; then
    echo "   FAIL: container start error" | tee -a "$RESULTS_FILE"
    echo "$start_out" | tail -5 | sed 's/^/   /' | tee -a "$RESULTS_FILE"
    RESULTS[$model]="FAIL-START"
    stop_server; continue
  fi

  echo "   waiting for /v1/models (up to ${TIMEOUT}s)..." | tee -a "$RESULTS_FILE"
  if ! wait_ready; then
    echo "   FAIL: timeout" | tee -a "$RESULTS_FILE"
    echo "   --- last logs ---" | tee -a "$RESULTS_FILE"
    docker logs --tail 15 vllm 2>&1 | sed 's/^/   /' | tee -a "$RESULTS_FILE"
    RESULTS[$model]="FAIL-TIMEOUT"
    stop_server; continue
  fi
  echo "   server ready" | tee -a "$RESULTS_FILE"

  kv_info=$(get_kv_info)
  echo "   KV: $kv_info" | tee -a "$RESULTS_FILE"
  KV_POOLS[$model]="$kv_info"
  CTX_USED[$model]="$target_ctx"

  echo "   running completion..." | tee -a "$RESULTS_FILE"
  if output=$(run_completion 2>&1); then
    echo "   PASS: $output" | tee -a "$RESULTS_FILE"
    RESULTS[$model]="PASS"
  else
    echo "   $output" | tee -a "$RESULTS_FILE"
    RESULTS[$model]="FAIL-COMPLETION"
  fi

  stop_server
done

# ── Summary ───────────────────────────────────────────────────────────────────

{
  echo ""
  echo "============================================================"
  echo " SUMMARY"
  echo "============================================================"
  printf "  %-22s  %-8s  %-8s  %s\n" "model" "ctx" "result" "kv_pool"
  printf "  %-22s  %-8s  %-8s  %s\n" "─────────────────────" "───────" "───────" "───────"
  for entry in "${ALL_MODELS[@]}"; do
    model="${entry%%|*}"
    rest="${entry#*|}"
    target_ctx="${rest%%|*}"
    status="${RESULTS[$model]:-SKIP}"
    kv="${KV_POOLS[$model]:-—}"
    printf "  %-22s  %-8s  %-8s  %s\n" "$model" "${CTX_USED[$model]:-$target_ctx}" "$status" "$kv"
  done
  echo "============================================================"
} | tee -a "$RESULTS_FILE"
