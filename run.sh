#!/usr/bin/env bash
# ─── vLLM launcher for the RTX 5090 ─────────────────────────────────────────
# Target hardware:
#   GPU: RTX 5090 (32 GB VRAM, Blackwell sm_120, native NVFP4/MXFP4 tensor cores)
#   CPU: any modern x86_64
#   RAM: 32 GB+ recommended
#
# vLLM runs inside a Docker container (image: vllm/vllm-openai:latest).
# Weights live in ./cache and are bind-mounted into the container so the
# server starts without touching HuggingFace at runtime. If the weights for
# the chosen model are missing, this script downloads them first.
#
# Exposes an OpenAI-compatible HTTP API on http://<HOST_IP>:8080/v1.
# Only one container can bind the port at a time.
#
# Each launch aliases the loaded model under many names (`default`, every
# model key below, plus the generic placeholders OpenAI clients tend to send)
# so any familiar model ID routes to whatever is currently booted. See
# SERVED_ALIASES below.
#
# ─── Usage ──────────────────────────────────────────────────────────────────
#   ./run.sh                           # pick a model (numbered menu), start server
#   ./run.sh <model>                   # start detached — tail with ./logs-vllm.sh
#   ./run.sh <model> -d                # same (always detached; -d kept for compat)
#   ./run.sh <model> [vllm args...]    # extra args forwarded to `vllm serve`
#   ./run.sh --help | -h               # this block
#   ./run.sh --list                    # list model keys only
#
# Stop a running container with ./stop-vllm.sh (docker rm -f vllm).
# Pull the latest image with ./update-vllm.sh.
#
# ─── Model lineup (measured on RTX 5090, vLLM 0.22.1) ───────────────────────
#
#   model              params         quant         ctx     tool-parser   notes
#   ─────────────────  ─────────────  ────────────  ──────  ────────────  ─────────────────
#   qwen36-27b-awq     27B dense      AWQ 4-bit     262K    qwen3_xml     ⭐ PREFERRED 27B, 2x decode vs nvfp4
#   qwen36-27b-nvfp4   27B dense      NVFP4         262K    qwen3_xml     Blackwell-native FP4
#   cascade2           30B/3B  MoE    NVFP4         131K    qwen3_coder   ⭐ Mamba2+attn, perfect tool, LiveCB 87.2%
#   qwen36             35B/3B  MoE    NVFP4         196K    qwen3_coder   vision + reasoning, fastest decode
#   qwen3-coder        30B/3B  MoE    AWQ 4-bit     221K    qwen3_coder   non-thinking coder specialist
#   gemma4             26B/4B  MoE    AWQ 4-bit     262K    gemma4        text+tool, 86.4% τ²-bench (mm disabled)
#   gemma4-coder       31B dense      NVFP4         262K    gemma4        ⭐ text-only coding daily-driver (alias: gemma4-text)
#   gpt-oss            21B/3.6B MoE   MXFP4         131K    openai        fastest; Reasoning: low|medium|high
#   nemotron3          31B/3B  MoE    NVFP4         224K    qwen3_coder   NVIDIA Omni, reasoning (mm disabled)
#
#   ctx = verified boot+completion ceiling on a single 32 GB 5090.
#
# ─── Picking one at a glance ────────────────────────────────────────────────
#   Best coding quality per token?          → qwen36-27b-awq (dense, 2x decode)
#   Fastest capable daily driver + vision?  → qwen36         (3B active MoE)
#   Coder tool-loop, predictable latency?   → qwen3-coder    (no thinking blocks)
#   Strong reasoning + perfect tool-use?    → cascade2       (Mamba2+attn MoE)
#   Gemma 4 text+tool+reasoning?            → gemma4         (MoE AWQ, mm disabled)
#   Gemma 4 dense coding, long context?     → gemma4-coder   (dense NVFP4, 262K)
#   OpenAI weights w/ reasoning dial?       → gpt-oss        ("Reasoning: high")
#   NVIDIA Omni reasoning MoE?              → nemotron3      (NVFP4, 224K)
#
# ─── Overrides ──────────────────────────────────────────────────────────────
#   Push context above the default:
#     ./run.sh qwen3-coder --max-model-len 262144
#   Free more VRAM for KV (turn memory util up):
#     ./run.sh qwen36 --gpu-memory-utilization 0.97
#   Bind a different host IP/port (default 0.0.0.0:8080):
#     HOST_IP=127.0.0.1 HOST_PORT=9090 ./run.sh gemma4
#   Extra vllm args:
#     ./run.sh qwen3-coder --max-num-seqs 64 --enable-chunked-prefill
#
# ─── Gotchas ────────────────────────────────────────────────────────────────
#   • Only one container at a time binds the port. Stop the previous one first.
#   • vLLM serves one model per container. To switch: ./stop-vllm.sh, then
#     ./run.sh <other>. Loading takes 30–120s depending on model size.
#   • Weights are bind-mounted from ./cache. Missing weights are downloaded
#     automatically; or fetch ahead of time with ./download-model.sh <repo>.
#   • The DeltaNet hybrids (qwen36 family) require prefix caching OFF and
#     --max-num-batched-tokens >= 4096 — both already set per-model below.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

IMAGE="vllm/vllm-openai:latest"
CONTAINER_NAME="vllm"
HOST_IP="${HOST_IP:-0.0.0.0}"
HOST_PORT="${HOST_PORT:-8080}"
CONTAINER_PORT=8000

# `--served-model-name` accepts multiple aliases — any of these names routes
# to whatever model is actually loaded. Keeps clients working without
# reconfiguration when switching models, and lets clients that hardcode a
# placeholder model ID (e.g. "gpt-4", "llama") continue to resolve.
SERVED_ALIASES=(
  default
  # This file's model keys:
  qwen36 qwen36-27b-nvfp4 qwen36-27b-awq qwen3-coder cascade2 gemma4 gemma4-text gemma4-coder gpt-oss nemotron3
  # Generic placeholders common OpenAI clients / agents default to. vLLM is
  # strict about the `model` field, so alias them to whatever is loaded.
  llama llama2 llama3 llama-3 chat model assistant local
  gpt gpt-3.5 gpt-3.5-turbo gpt-4 gpt-4o gpt-5
  claude claude-3 claude-3.5 claude-opus claude-sonnet
  qwen gemma deepseek mistral
)

# Flags shared across all model launches. Per-model blocks may append more.
COMMON_ARGS=(
  --host 0.0.0.0
  --port "$CONTAINER_PORT"
  --served-model-name "${SERVED_ALIASES[@]}"
  --gpu-memory-utilization 0.92
  --dtype auto
  --trust-remote-code
  --enable-chunked-prefill
  --enable-prefix-caching
  --max-num-seqs 64
)

usage() {
  # print the leading comment block (skip the shebang)
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
  exit 0
}

# Interactive picker, shown when run.sh is invoked with no arguments.
# Order here doubles as the "1-N" numbering shown to the user.
MODELS=(
  "qwen36-27b-awq|27B dense AWQ-INT4 (cyankiwi), 262K — ⭐ PREFERRED 27B: best quality/token, 2x faster decode"
  "qwen36-27b-nvfp4|27B dense NVFP4 (sakamakismile), 262K — Blackwell-native FP4, 27B dense"
  "cascade2|30B/3B MoE NVFP4 (chankhavu), 131K — ⭐ Mamba2+attn, perfect tool-use, LiveCodeBench 87.2%"
  "qwen36|35B/3B MoE NVFP4 (RedHatAI), 196K, vision — newest Qwen flagship MoE, fastest capable decode"
  "qwen3-coder|30B/3B MoE AWQ (cyankiwi), 221K — non-thinking coder specialist, ~277 t/s"
  "gemma4|26B/4B MoE AWQ (cyankiwi), 262K — text+tool, mm disabled (86.4% τ²-bench)"
  "gemma4-coder|31B dense NVFP4 (LilaRest), 262K — ⭐ text-only coding daily-driver, no mm overhead"
  "gpt-oss|21B/3.6B MoE MXFP4 (openai), 131K — ⭐ OpenAI small, Reasoning: low/med/high"
  "nemotron3|31B/3B MoE NVFP4 (NVIDIA Omni), 224K — reasoning, mm disabled, text-only"
)

pick_model() {
  if ! { exec 9<>/dev/tty; } 2>/dev/null; then
    {
      echo "run.sh: no TTY available — pass a model name, or use -h for help."
      echo "Available models:"
      for entry in "${MODELS[@]}"; do
        echo "  ${entry%%|*}"
      done
    } >&2
    exit 1
  fi
  echo "Select a model:" >&2
  local i=1
  for entry in "${MODELS[@]}"; do
    local id="${entry%%|*}"
    local desc="${entry#*|}"
    printf "  %d) %-16s %s\n" "$i" "$id" "$desc" >&2
    i=$((i + 1))
  done
  echo >&2
  local choice
  read -r -u 9 -p "Choice [1-${#MODELS[@]}] (q to quit): " choice
  exec 9<&-
  case "$choice" in
    q|Q|"") echo "Cancelled." >&2; exit 0 ;;
  esac
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#MODELS[@]} )); then
    echo "Invalid choice: $choice" >&2
    exit 1
  fi
  local entry="${MODELS[$((choice - 1))]}"
  printf '%s' "${entry%%|*}"
}

list_only() {
  for entry in "${MODELS[@]}"; do echo "${entry%%|*}"; done
  exit 0
}

# Resolve model name → MODEL_ARGS array + SNAPSHOT_REPO (+ any EXTRA_ENV/VOLS).
select_model() {
  case "$1" in
    qwen36)
      # RedHatAI/Qwen3.6-35B-A3B-NVFP4 (~25 GB weights). Alibaba's newest Qwen
      # flagship MoE: hybrid Gated DeltaNet + Gated Attention, 35B total / 3B
      # active, native vision. NVFP4 via NVIDIA Model Optimizer runs natively on
      # the 5090's FP4 tensor cores (no dequant step).
      #
      # 25 GB weights is TIGHT on 32 GB — ~6 GB headroom for KV. DeltaNet hybrid
      # needs --max-num-batched-tokens >= block_size (vLLM default 2048 trips an
      # AssertionError). --no-enable-prefix-caching: DeltaNet layers carry a
      # recurrent state not reflected in KV blocks, so caching KV produces wrong
      # outputs. --max-num-seqs 1: KV headroom can't support concurrent seqs.
      # ctx 196K: 16 full-attention layers × fp8 KV ≈ 6.3 GB; 262K would need
      # ~8.4 GB which exceeds headroom.
      SNAPSHOT_REPO="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
      MODEL_ARGS=(
        --max-model-len 196608
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.95
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
        --reasoning-parser qwen3
      )
      ;;
    qwen3-coder)
      # cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit (~18 GB weights).
      # 30B / 3B active MoE, 262K native ctx. AWQ INT4 (group_size=32, better
      # quality than typical 128). Tool parser qwen3_coder (canonical for this
      # family). No reasoning parser (Instruct = no thinking blocks).
      # NOTE: packed as compressed-tensors (not awq_marlin) — let vLLM
      # auto-detect by omitting --quantization.
      # ctx 221K: 262144 OOMs on KV allocation (needs 12 GiB KV, only ~11 GiB
      # available after weights). 221184 is the confirmed ceiling.
      SNAPSHOT_REPO="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
      MODEL_ARGS=(
        --max-model-len 221184
        --kv-cache-dtype fp8
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
      )
      ;;
    qwen36-27b-nvfp4)
      # sakamakismile/Qwen3.6-27B-NVFP4 (~19.7 GB weights). NVFP4 via NVIDIA
      # Model Optimizer with Blackwell sm_120 GEMM kernels, vision tower in BF16.
      # Tool parser: qwen3_xml works better than qwen3_coder for the 27B dense
      # variant (qwen3_coder was tuned for the Coder model).
      # ctx 262K: native ceiling. KV grows only on 16 full-attention layers
      # (DeltaNet layers have fixed recurrent state, no KV).
      SNAPSHOT_REPO="sakamakismile/Qwen3.6-27B-NVFP4"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.95
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --enable-chunked-prefill
        --enable-auto-tool-choice
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
      )
      ;;
    qwen36-27b-awq)
      # cyankiwi/Qwen3.6-27B-AWQ-INT4 (~20.4 GB). Same Qwen3.6-27B base as the
      # NVFP4 variant but calibrated AWQ-INT4 (group_size=32). Essentially the
      # same quality as NVFP4 but ~2x faster decode → PREFERRED 27B option.
      # ctx 262K: native ceiling; KV only on 16 full-attention layers.
      SNAPSHOT_REPO="cyankiwi/Qwen3.6-27B-AWQ-INT4"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.95
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --enable-auto-tool-choice
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
      )
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    nemotron3)
      # nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 (~20.9 GB weights).
      # 31B / ~3B active hybrid Mamba2 + attention MoE, NVFP4 with FP8 block
      # scale. "Omni" adds multimodal input; "Reasoning" adds chain-of-thought
      # via the nemotron_v3 parser. Multimodal is disabled here for a text-only
      # focus (avoids encoder profiling OOM).
      # ctx 229K: 262K OOMs even with mm disabled (vLLM 0.22.1 overhead).
      # util 0.95 (not 0.97): the FlashInfer fp8_gemm AutoTuner needs ~132 MiB
      # temp buffers during init; at 0.97 only 89 MiB free → OOM.
      SNAPSHOT_REPO="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
      MODEL_ARGS=(
        --quantization modelopt_fp4
        --max-model-len 229376
        --max-num-seqs 1
        --gpu-memory-utilization 0.95
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
        --reasoning-parser nemotron_v3
      )
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    cascade2)
      # chankhavu/Nemotron-Cascade-2-30B-A3B-NVFP4 (~19 GB on disk). NVIDIA's
      # hybrid Mamba2 + attention MoE, 30B / 3B active. Strong reasoning +
      # perfect tool-use. Base benchmarks (BF16): LiveCodeBench v6 87.2,
      # SWE-V 50.2, GPQA-Diamond 76.1, AIME25 92.4.
      # Quant must be specified explicitly (modelopt_fp4, not auto-detected).
      # ctx 131K: Mamba state is fixed-size (no KV growth); only attention
      # layers grow KV. ~19 GB weights + 0.94 util leaves ~11 GB KV headroom.
      SNAPSHOT_REPO="chankhavu/Nemotron-Cascade-2-30B-A3B-NVFP4"
      MODEL_ARGS=(
        --quantization modelopt_fp4
        --max-model-len 131072
        --max-num-seqs 1
        --gpu-memory-utilization 0.94
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
        --reasoning-parser nemotron_v3
      )
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    gemma4)
      # cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (~15 GB weights). AWQ INT4 of
      # Gemma 4 26B/4B MoE. Multimodal disabled via image=1,audio=0 to suppress
      # the encoder profiling that OOMs the NVFP4 variant.
      # Chat template mounted from ./templates/gemma4-tool-template.jinja
      # (required for correct Gemma 4 pythonic tool-call formatting).
      # util 0.94: vLLM's CUDA-graph profiling + AWQ Marlin MoE kernel need
      # ~44 MiB intermediate buffers; 0.97 OOMs.
      SNAPSHOT_REPO="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.94
        --kv-cache-dtype fp8
        --limit-mm-per-prompt '{"image":1,"audio":0}'
        --enable-auto-tool-choice
        --tool-call-parser gemma4
        # --reasoning-parser gemma4 DISABLED: vLLM bug — Gemma4's SentencePiece
        # tokenizer fails to unpickle during EngineCore IPC init. Re-enable when
        # fixed upstream.
        --chat-template /gemma4-tool-template.jinja
      )
      EXTRA_VOLS+=(-v "$SCRIPT_DIR/templates/gemma4-tool-template.jinja:/gemma4-tool-template.jinja")
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    gemma4-text|gemma4-coder)
      # LilaRest/gemma-4-31B-it-NVFP4-turbo (~18.5 GB). Text-only fork of Gemma 4
      # 31B dense: video + audio encoders stripped, uses Gemma4ForCausalLM (not
      # ForConditionalGeneration) — avoids all multimodal profiling OOMs. The
      # strongest-reasoning Gemma 4 for the 5090 → the dense coding daily-driver.
      # ctx 262K (full native): Gemma 4 interleaves attention 5:1 — only 10 of 60
      # layers are full-attention (KV grows with context); the other 50 are
      # sliding-window (capped at 1024 tokens, fixed KV). So long-context KV is
      # ~80 KB/token at fp8 (the 10 global layers), NOT 480 KB/token — ~6x
      # cheaper than a fully global model. With --max-num-seqs 1 the whole KV
      # budget feeds one sequence; at util 0.975 the KV pool holds ~272K tokens,
      # enough for the model's full 262144 ceiling (1.04x). Verified on the 5090:
      # booted + a 200K-token needle-in-haystack prompt recalled correctly.
      # util is 0.975, NOT 0.98: 0.98 grabs so much KV that the warmup forward
      # pass OOMs (needs ~84 MiB free); 0.975 leaves room. If a driver/vLLM
      # update tips it into warmup OOM, fall back to 0.97 + --max-model-len
      # 245760 (~240K, comfortable margin).
      SNAPSHOT_REPO="LilaRest/gemma-4-31B-it-NVFP4-turbo"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-seqs 1
        --gpu-memory-utilization 0.975
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --enable-auto-tool-choice
        --tool-call-parser gemma4
        # --reasoning-parser gemma4 DISABLED: same vLLM tokenizer bug as above.
        --chat-template /gemma4-tool-template.jinja
      )
      EXTRA_VOLS+=(-v "$SCRIPT_DIR/templates/gemma4-tool-template.jinja:/gemma4-tool-template.jinja")
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    gpt-oss)
      # openai/gpt-oss-20b (~13.8 GB safetensors, MXFP4-native MoE weights).
      # OpenAI's open-weight 21B total / 3.6B active MoE. MXFP4 runs natively on
      # Blackwell tensor cores. Reasoning is configurable by prefixing the system
      # prompt with "Reasoning: low|medium|high" — NOT a vLLM flag.
      # Tool parser openai; reasoning parser openai_gptoss extracts the
      # harmony-tagged chain-of-thought into the reasoning field.
      # 131K IS THE CEILING: config has rope_scaling YaRN factor=32 from a 4K
      # base — already at OpenAI's tested envelope. Pushing max-model-len higher
      # crashes the container with a CUDA device-side assert on long prompts.
      # stock openai/gpt-oss-20b IS the optimal checkpoint for sm_120: weights
      # are MXFP4-native; NVFP4 re-quants give nothing over it, BF16/GGUF are for
      # other runtimes. Do NOT set VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 — that
      # TRT-LLM MXFP4xMXFP8 MoE kernel is B200/sm_100 only and FAILS to boot on
      # the 5090 (sm_120); the default 'auto' MoE backend is correct.
      # SPEED: EAGLE3 speculative decoding is a measured win at batch 1 — but
      # only when tuned: num_speculative_tokens 3 (not 7) + --max-num-batched-
      # tokens 8192 (the default cap of 2048 under spec throttles it). Verified
      # on the 5090: ~293 t/s → ~352 prose / ~415 structured (1.2–1.4x), lossless
      # (same outputs), tool-calling still faithful. num_speculative_tokens 7
      # was SLOWER than no-spec (low accept rate + scheduler cap). The eagle3
      # head (~0.3 GB) auto-downloads via SPECULATOR_REPO. If a vLLM/driver
      # update breaks the drafter, drop --speculative-config to fall back clean.
      SNAPSHOT_REPO="openai/gpt-oss-20b"
      SPECULATOR_REPO="RedHatAI/gpt-oss-20b-speculator.eagle3"
      MODEL_ARGS=(
        --max-model-len 131072
        --kv-cache-dtype auto
        --max-num-batched-tokens 8192
        --speculative-config '{"model":"RedHatAI/gpt-oss-20b-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3"}'
        --enable-auto-tool-choice
        --tool-call-parser openai
        --reasoning-parser openai_gptoss
      )
      ;;
    *)
      echo "Unknown model: $1" >&2
      echo "Run '$0 --help' to see available models (or --list for just keys)." >&2
      exit 1
      ;;
  esac
}

# Resolve a HuggingFace repo to its actual snapshot path inside ./cache. If the
# weights are missing, download them first (one-click). We bind the whole cache
# tree into the container so the snapshot/<rev>/ symlinks into blobs/ resolve.
resolve_snapshot() {
  local repo="$1"
  local dirname="models--${repo//\//--}"
  local cache_root="$SCRIPT_DIR/cache/$dirname/snapshots"
  if [[ ! -d "$cache_root" ]]; then
    echo ">>> weights for $repo not found in cache — downloading now..." >&2
    "$SCRIPT_DIR/download-model.sh" "$repo" >&2
  fi
  if [[ ! -d "$cache_root" ]]; then
    echo "run.sh: snapshot dir still missing after download — $cache_root" >&2
    echo "       Try manually: ./download-model.sh $repo" >&2
    exit 1
  fi
  # Pick the most recent snapshot (there's usually only one).
  local snap
  snap=$(ls -1 "$cache_root" | head -1)
  [[ -n "$snap" ]] || { echo "run.sh: no snapshot inside $cache_root" >&2; exit 1; }
  printf '%s' "$cache_root/$snap"
}

# ─── Dispatch ────────────────────────────────────────────────────────────
case "${1:-}" in
  -h|--help) usage ;;
  --list)    list_only ;;
esac

target=""
case "${1:-}" in
  "")   target="$(pick_model)" ;;
  *)    target="$1"; shift ;;
esac
[ -n "$target" ] || { echo "No model selected." >&2; exit 1; }

EXTRA_ENV=()
EXTRA_VOLS=()
MODEL_ARGS=()
SNAPSHOT_REPO=""
SPECULATOR_REPO=""   # optional EAGLE3/draft repo; downloaded to cache and resolved by id
select_model "$target"

# Always launch detached. Tolerate a leading `-d` for backward compat.
if [[ "${1:-}" == "-d" ]]; then
  shift
fi

# HF cache snapshot/ entries are symlinks into ../../blobs/<hash>, so we must
# bind-mount the whole cache/ tree (not just the snapshot dir).
SNAPSHOT_HOST="$(resolve_snapshot "$SNAPSHOT_REPO")"
SNAPSHOT_REL="${SNAPSHOT_HOST#$SCRIPT_DIR/cache/}"
SNAPSHOT_CONTAINER="/root/.cache/huggingface/$SNAPSHOT_REL"

# Speculative-decoding draft head (if the model sets one): ensure it's in the
# mounted cache so vLLM resolves it by repo id offline, like the main weights.
[[ -n "$SPECULATOR_REPO" ]] && resolve_snapshot "$SPECULATOR_REPO" >/dev/null

# Stop any previous container first; only one binds the port.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

mkdir -p "$SCRIPT_DIR/logs"

RUN_ARGS=(
  --name "$CONTAINER_NAME"
  --runtime nvidia
  --gpus all
  --ipc host
  -p "${HOST_IP}:${HOST_PORT}:${CONTAINER_PORT}"
  -v "$SCRIPT_DIR/cache:/root/.cache/huggingface"
  -v "$SCRIPT_DIR/logs:/logs"
  -e "HF_HUB_CACHE=/root/.cache/huggingface"
  "${EXTRA_ENV[@]}"
  "${EXTRA_VOLS[@]}"
)

# Always detached. --restart no: if init fails, don't flap.
RUN_ARGS+=(-d --restart no)

display_ip="$HOST_IP"
[[ "$display_ip" == "0.0.0.0" ]] && display_ip="localhost"

echo ">>> model       : $target"
echo ">>> snapshot    : $SNAPSHOT_HOST"
echo ">>> endpoint    : http://${display_ip}:${HOST_PORT}/v1"
echo ">>> served name : default (+ aliases)"
echo ">>> cli args    : ${MODEL_ARGS[*]} ${COMMON_ARGS[*]} $*"

# Order matters: COMMON_ARGS first (shared defaults), MODEL_ARGS second
# (per-model overrides), "$@" last (user overrides everything). vLLM's argparse
# takes the last-wins value for repeated flags.
docker run "${RUN_ARGS[@]}" "$IMAGE" \
  "$SNAPSHOT_CONTAINER" \
  "${COMMON_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "$@"

echo ">>> started detached — tail with ./logs-vllm.sh, stop with ./stop-vllm.sh"
