#!/usr/bin/env bash
# Pull the latest vllm/vllm-openai image.
set -euo pipefail

echo ">>> pulling vllm/vllm-openai:latest..."
docker pull vllm/vllm-openai:latest
echo ">>> pruning dangling vllm images..."
docker image prune -f --filter "label=org.opencontainers.image.source=https://github.com/vllm-project/vllm" 2>/dev/null || true
echo ">>> current image:"
docker images vllm/vllm-openai --format 'table {{.Repository}}:{{.Tag}}\t{{.CreatedSince}}\t{{.Size}}' | head
