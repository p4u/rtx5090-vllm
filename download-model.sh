#!/usr/bin/env bash
# Download a HuggingFace repo into the vLLM cache directory so the container
# can load it offline.
#
# Usage:
#   ./download-model.sh <user/repo>              # all files
#   ./download-model.sh <user/repo> <glob>       # matching files only
#   ./download-model.sh --all                    # download every model in run.sh
#
# Tips:
#   - The `hf` CLI is preferred (pip install huggingface_hub[hf_xet]); a
#     Docker fallback is used if it isn't on PATH.
#   - Export HF_TOKEN=hf_... to avoid unauthenticated rate limits.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CACHE_DIR="$SCRIPT_DIR/cache"
mkdir -p "$CACHE_DIR"

# Every repo referenced by run.sh, keyed by its run.sh model name.
# Keep this list in sync with the SNAPSHOT_REPO values in run.sh.
DEFAULT_REPOS=(
  "cyankiwi/Qwen3.6-27B-AWQ-INT4"                          # qwen36-27b-awq  (preferred 27B)
  "sakamakismile/Qwen3.6-27B-NVFP4"                        # qwen36-27b-nvfp4
  "chankhavu/Nemotron-Cascade-2-30B-A3B-NVFP4"             # cascade2
  "RedHatAI/Qwen3.6-35B-A3B-NVFP4"                         # qwen36
  "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"        # qwen3-coder
  "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"                  # gemma4
  "LilaRest/gemma-4-31B-it-NVFP4-turbo"                    # gemma4-text
  "openai/gpt-oss-20b"                                     # gpt-oss
  "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"   # nemotron3
)

download_one() {
    local repo="$1"
    local pattern="${2:-}"
    echo ">>> downloading $repo${pattern:+ (glob: $pattern)}"

    if command -v hf &>/dev/null; then
        if [[ -n "$pattern" ]]; then
            hf download "$repo" --include "$pattern" --cache-dir "$CACHE_DIR"
        elif [[ "$repo" == "openai/gpt-oss-20b" ]]; then
            # Skip the metal/ and original/ duplicates (~27 GB extra).
            hf download "$repo" --exclude "metal/*" --exclude "original/*" --cache-dir "$CACHE_DIR"
        else
            hf download "$repo" --cache-dir "$CACHE_DIR"
        fi
    else
        echo ">>> hf CLI not found, using docker method..."
        docker run --rm -v "$CACHE_DIR:/cache" \
            -e "HF_HOME=/cache" \
            ${HF_TOKEN:+-e "HF_TOKEN=$HF_TOKEN"} \
            python:3.12-slim bash -c "
                pip install -q huggingface-hub[hf_xet] && \
                hf download $repo ${pattern:+--include $pattern} --cache-dir /cache
            "
    fi
}

if [[ "${1:-}" == "--all" ]]; then
    for repo in "${DEFAULT_REPOS[@]}"; do
        download_one "$repo"
    done
    echo ">>> all downloads complete"
    exit 0
fi

if [[ -z "${1:-}" ]]; then
    echo "usage: $0 <user/repo> [include-glob]" >&2
    echo "       $0 --all    # download every model configured in run.sh" >&2
    exit 1
fi

download_one "$1" "${2:-}"
