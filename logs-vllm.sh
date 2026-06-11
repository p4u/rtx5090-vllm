#!/usr/bin/env bash
# Tail logs from the running vLLM container.
set -euo pipefail
exec docker logs -f --tail 200 vllm
