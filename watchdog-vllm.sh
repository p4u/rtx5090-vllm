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
# Env:
#   CONTAINER_NAME   container to watch (default: vllm)
#   FAILS_BEFORE     consecutive unhealthy checks before restarting (default: 2)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME="${CONTAINER_NAME:-vllm}"
FAILS_BEFORE="${FAILS_BEFORE:-2}"
STATE_FILE="${TMPDIR:-/tmp}/vllm-watchdog.$NAME.fails"
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
  # healthy / starting / none → reset the counter, nothing to do
  rm -f "$STATE_FILE" 2>/dev/null || true
  echo "$(ts) [watchdog] '$NAME' state=$state health=$health — ok"
fi
exit 0
