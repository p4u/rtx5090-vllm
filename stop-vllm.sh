#!/usr/bin/env bash
# Gracefully stop the running vLLM container.
set -euo pipefail

if docker ps -a --format '{{.Names}}' | grep -qx vllm; then
  docker rm -f vllm
  echo "vllm stopped"
else
  echo "vllm is not running"
fi
