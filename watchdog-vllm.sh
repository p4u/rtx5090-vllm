#!/usr/bin/env bash
# Restart the vLLM container if it hangs (alive but unresponsive).
#
# Docker's --restart policy only fires when the process EXITS. A vLLM server
# that deadlocks (CUDA hang, stuck request) stays "running" forever, so the
# HEALTHCHECK marks it `unhealthy` but nothing acts on it. This watchdog closes
# that gap: run it periodically (cron or a systemd timer) and it restarts the
# container the moment it's been unhealthy long enough.
#
# One-shot by design — schedule it, don't loop it. Examples:
#   crontab:   * * * * * /home/p4u/rtx5090-vllm/watchdog-vllm.sh >> /home/p4u/rtx5090-vllm/logs/watchdog.log 2>&1
#   systemd:   a vllm-watchdog.service (Type=oneshot) + vllm-watchdog.timer (OnUnitActiveSec=60s)
#
# It also detects LIVELOCKED inference (seen in the field 2026-09-05 on
# qwen38-fast: /health kept answering 200 while one request held the engine
# for 5+ hours at 99% GPU / 460W producing ~zero tokens — with
# --max-num-seqs 1 that blocks every other request, and the healthcheck
# never trips). Detection: requests are running but the prompt+generation
# token counters in /metrics have advanced less than STALL_MIN_TOKENS since
# a snapshot older than STALL_AFTER_S. Normal decode moves thousands of
# tokens per minute; the deepest legitimate prefill moves the prompt counter
# continuously — a frozen pair with running>0 means a wedged engine.
#
# Env:
#   CONTAINER_NAME    container to watch (default: vllm)
#   FAILS_BEFORE      consecutive unhealthy checks before restarting (default: 2)
#   METRICS_URL       override the /metrics endpoint (default: derived from
#                     the container's published port; https handled). Both
#                     engines are understood: vLLM's vllm:* counters and
#                     llama-server's llamacpp:* ones. The scrape authenticates
#                     with the container's own VLLM_API_KEY/LLAMA_API_KEY,
#                     which llama-server requires for /metrics.
#   STALL_AFTER_S     seconds of near-zero progress with running>0 before
#                     restarting (default: 600)
#   STALL_MIN_TOKENS  progress below this over the window counts as stalled
#                     (default: 60)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME="${CONTAINER_NAME:-vllm}"
FAILS_BEFORE="${FAILS_BEFORE:-2}"
STALL_AFTER_S="${STALL_AFTER_S:-600}"
STALL_MIN_TOKENS="${STALL_MIN_TOKENS:-60}"
STATE_FILE="${TMPDIR:-/tmp}/vllm-watchdog.$NAME.fails"
STALL_FILE="${TMPDIR:-/tmp}/vllm-watchdog.$NAME.stall"
ts() { date -Is; }

# Container state: running/exited/missing, plus health if present.
state=$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null || echo "missing")
health=$(docker inspect "$NAME" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>/dev/null || echo "none")

if [[ "$state" == "missing" ]]; then
  # Never launched, or removed on purpose — nothing to heal. Don't recreate;
  # relaunching needs a model choice, which is ./run.sh's job.
  echo "$(ts) [watchdog] container '$NAME' missing — nothing to do"
  rm -f "$STATE_FILE"
  exit 0
fi

# If Docker's restart policy is doing its job (exited→restarting), let it. Only
# a hung *running* container needs us.
if [[ "$state" != "running" ]]; then
  echo "$(ts) [watchdog] '$NAME' state=$state (Docker restart policy handles exits) — leaving it"
  rm -f "$STATE_FILE"
  exit 0
fi

if [[ "$health" == "unhealthy" ]]; then
  fails=$(( $(cat "$STATE_FILE" 2>/dev/null || echo 0) + 1 ))
  echo "$fails" > "$STATE_FILE"
  echo "$(ts) [watchdog] '$NAME' unhealthy ($fails/$FAILS_BEFORE)"
  if (( fails >= FAILS_BEFORE )); then
    echo "$(ts) [watchdog] restarting hung '$NAME'"
    docker restart "$NAME" >/dev/null 2>&1 && echo "$(ts) [watchdog] restarted" || echo "$(ts) [watchdog] restart FAILED"
    rm -f "$STATE_FILE"
  fi
else
  # healthy / starting / none → reset the counter
  rm -f "$STATE_FILE" 2>/dev/null || true

  # ── livelock check (healthy but not making progress) ──
  if [[ -z "${METRICS_URL:-}" ]]; then
    hostport=$(docker inspect "$NAME" --format \
      '{{with index .HostConfig.PortBindings "8000/tcp"}}{{(index . 0).HostIp}}:{{(index . 0).HostPort}}{{end}}' 2>/dev/null || true)
    hostport="${hostport/0.0.0.0/127.0.0.1}"
    METRICS_URL="http://${hostport:-127.0.0.1:8080}/metrics"
  fi
  # vLLM leaves /metrics unauthenticated, but llama-server puts it behind the
  # same --api-key as /v1 — without this header the scrape 401s and every
  # livelock looks like an idle server (running=0 → "ok"), which is exactly the
  # blind spot this check exists to close. Read the key from the container's
  # own Env, like the UI proxy does, so token renewal can't desync it.
  api_key=$(docker inspect "$NAME" --format \
    '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
    | sed -n 's/^\(VLLM_API_KEY\|LLAMA_API_KEY\)=//p' | head -n1)
  metrics=$(curl -sk --max-time 5 ${api_key:+-H "Authorization: Bearer $api_key"} "$METRICS_URL" 2>/dev/null \
            || curl -sk --max-time 5 ${api_key:+-H "Authorization: Bearer $api_key"} "${METRICS_URL/http:/https:}" 2>/dev/null || true)
  # Two metric namespaces, one meaning: vLLM exports vllm:*, llama-server
  # (RUNTIME=llamacpp models) exports llamacpp:* with different spellings for
  # the same three counters.
  running=$(printf '%s' "$metrics" | awk \
    '/^vllm:num_requests_running|^llamacpp:requests_processing/{s+=$2} END{printf "%d", s}')
  tokens=$(printf '%s' "$metrics" | awk \
    '/^vllm:prompt_tokens_total|^vllm:generation_tokens_total|^llamacpp:prompt_tokens_total|^llamacpp:tokens_predicted_total/{s+=$2} END{printf "%d", s}')
  if [[ -z "$metrics" || "$running" -eq 0 ]]; then
    rm -f "$STALL_FILE"
    echo "$(ts) [watchdog] '$NAME' health=$health running=${running:-?} — ok"
    exit 0
  fi
  now=$(date +%s)
  if [[ -f "$STALL_FILE" ]]; then
    read -r snap_time snap_tokens < "$STALL_FILE" || { snap_time=$now; snap_tokens=$tokens; }
    progress=$(( tokens - snap_tokens ))
    age=$(( now - snap_time ))
    if (( progress >= STALL_MIN_TOKENS )); then
      echo "$now $tokens" > "$STALL_FILE"   # progressing — new snapshot
      echo "$(ts) [watchdog] '$NAME' ok (running=$running, +$progress tokens in ${age}s)"
    elif (( age >= STALL_AFTER_S )); then
      echo "$(ts) [watchdog] '$NAME' LIVELOCKED: running=$running but only +$progress tokens in ${age}s — restarting"
      docker restart "$NAME" >/dev/null 2>&1 && echo "$(ts) [watchdog] restarted" || echo "$(ts) [watchdog] restart FAILED"
      rm -f "$STALL_FILE"
    else
      echo "$(ts) [watchdog] '$NAME' low progress (+$progress tokens in ${age}s/${STALL_AFTER_S}s) — watching"
    fi
  else
    echo "$now $tokens" > "$STALL_FILE"
    echo "$(ts) [watchdog] '$NAME' running=$running — snapshot taken"
  fi
fi
exit 0
