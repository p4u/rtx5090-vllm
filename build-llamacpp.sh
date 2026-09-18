#!/usr/bin/env bash
# Build the llama.cpp image used by RUNTIME=llamacpp models (see run.sh).
#
# Why this exists: Bonsai's ternary weights need prism-ml's llama.cpp fork —
# stock llama.cpp rejects PQ2_0/PTQ1_0, and vLLM cannot read them at all (ggml
# type id 142 plus a Hadamard activation transform live only in the fork).
# prism-ml publishes no Docker image, so we build one.
#
# Usage:
#   ./build-llamacpp.sh                 # detect this GPU's compute capability
#   CUDA_ARCH=90 ./build-llamacpp.sh    # build for a different card (H100)
#   LLAMACPP_IMAGE=foo:bar ./build-llamacpp.sh
#   FORK_REF=prism ./build-llamacpp.sh  # branch/tag/commit of the fork
#
# Takes ~20 min: it compiles CUDA kernels for one architecture.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

IMAGE="${LLAMACPP_IMAGE:-bonsai-llamacpp:prism}"
FORK_URL="${FORK_URL:-https://github.com/PrismML-Eng/llama.cpp}"
# Branch `prism` is the maintained line behind the Bonsai models. Do NOT build
# prism-v6 (stale mid-migration) or prism-v5 (frozen, legacy format only).
FORK_REF="${FORK_REF:-prism}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/.build/llamacpp}"

# CUDA arch: compile for THIS host's GPU only — building every architecture
# multiplies compile time for kernels the card will never run. compute_cap
# comes back as e.g. "12.0" (Blackwell sm_120) → CMake wants "120".
if [[ -z "${CUDA_ARCH:-}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "build-llamacpp.sh: nvidia-smi not found — set CUDA_ARCH=<nn> explicitly" >&2
    exit 1
  fi
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
  [[ -n "$cap" ]] || { echo "build-llamacpp.sh: could not read compute_cap — set CUDA_ARCH" >&2; exit 1; }
  CUDA_ARCH="${cap//./}"
fi

echo ">>> image     : $IMAGE"
echo ">>> fork      : $FORK_URL ($FORK_REF)"
echo ">>> cuda arch : sm_${CUDA_ARCH}"

mkdir -p "$(dirname "$WORK_DIR")"
if [[ -d "$WORK_DIR/.git" ]]; then
  echo ">>> updating existing checkout"
  git -C "$WORK_DIR" fetch --depth 1 origin "$FORK_REF"
  git -C "$WORK_DIR" checkout -q FETCH_HEAD
else
  rm -rf "$WORK_DIR"
  git clone --depth 1 --branch "$FORK_REF" "$FORK_URL" "$WORK_DIR"
fi
echo ">>> commit    : $(git -C "$WORK_DIR" log -1 --format='%h %ad %s' --date=short)"

# --target server: the image's entrypoint is llama-server, so run.sh passes
# server flags straight through as the container command.
docker build \
  --target server \
  --build-arg CUDA_DOCKER_ARCH="$CUDA_ARCH" \
  -t "$IMAGE" \
  -f "$WORK_DIR/.devops/cuda.Dockerfile" \
  "$WORK_DIR"

echo ">>> built $IMAGE"
docker images "$IMAGE" --format '>>> size      : {{.Size}}'
