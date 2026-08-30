#!/usr/bin/env bash
# Stop and remove the vllm-ui container (the served model keeps running).
set -euo pipefail
if docker ps -a --format '{{.Names}}' | grep -qx vllm-ui; then
  docker rm -f vllm-ui >/dev/null
  echo "vllm-ui stopped"
else
  echo "vllm-ui is not running"
fi
