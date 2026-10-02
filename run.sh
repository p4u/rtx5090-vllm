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
# ─── Model lineup (measured on RTX 5090) ────────────────────────────────────
# Image is vllm/vllm-openai:latest (currently 0.28.0; qwen38-27b REQUIRES
# >= 0.28 — the Qwen3.8 arch is missing before it). The two unsloth qwen36
# entries REQUIRE vLLM >= 0.24 (quantized lm_head). NOTE: the LilaRest text-only
# Gemma 4 (gemma4-coder) was REMOVED — its quantized lm_head breaks the gemma4.py
# tie_weights() path on vLLM >= 0.24; gemma4-vision (unquantized lm_head) replaces
# it. Other entries were first verified on 0.22.1 — re-check a model if its
# numbers look off after an image pull.
#
# Most entries run on that vLLM image. bonsai2 is the exception: RUNTIME=llamacpp
# launches it on a locally-built prism-ml llama.cpp fork (./build-llamacpp.sh),
# the only engine that can execute its ternary tensors — see its case block.
#
#   model              params         quant         ctx     tool-parser   notes
#   ─────────────────  ─────────────  ────────────  ──────  ────────────  ─────────────────
#   bonsai2            27B dense+vis  ternary PQ2_0 262K    jinja         ⭐ FASTEST: ~141 t/s, 6.8 GB weights [llama.cpp fork, NOT vLLM]
#   qwen38-27b         27B dense      NVFP4-dyn     262K    qwen3_xml     ⭐ Qwen3.8 QUALITY flavor (mm off) [needs vLLM>=0.28]
#   qwen38-fast        27B dense      NVFP4+MTP     262K    qwen3_xml     Qwen3.8 SPEED flavor, ~44.7 t/s spec decode (mm off) [>=0.28]
#   qwen38-vision      27B dense+vis  NVFP4         131K    qwen3_xml     Qwen3.8 VISION flavor, image input [>=0.28]
#   qwen36-27b-awq     27B dense      AWQ 4-bit     262K    qwen3_xml     ⭐ PREFERRED 27B, 2x decode vs nvfp4
#   qwen36-27b-nvfp4   27B dense      NVFP4         262K    qwen3_xml     Blackwell-native FP4
#   qwen36-27b-unsloth 27B dense      NVFP4-dyn     262K    qwen3_xml     unsloth dynamic NVFP4, higher-q (mm off) [needs vLLM>=0.24]
#   qwen36-fast        35B/3B  MoE    NVFP4-dyn     262K    qwen3_coder   unsloth 35B-A3B, thinks heavily (mm off) [needs vLLM>=0.24]
#   cascade2           30B/3B  MoE    NVFP4         131K    qwen3_coder   ⭐ Mamba2+attn, perfect tool, LiveCB 87.2%
#   qwen36             35B/3B  MoE    NVFP4         196K    qwen3_coder   vision + reasoning, fastest decode
#   qwen3-coder        30B/3B  MoE    AWQ 4-bit     221K    qwen3_coder   non-thinking coder specialist
#   gemma4             26B/4B  MoE    AWQ 4-bit     262K    gemma4        text+tool, 86.4% τ²-bench (mm disabled)
#   gemma4-vision      31B dense+vis  NVFP4         128K    gemma4        vision+reasoning Gemma 4, ~69 t/s (verified 0.25.1)
#   gpt-oss            21B/3.6B MoE   MXFP4         131K    openai        fastest; Reasoning: low|medium|high
#   nemotron3          31B/3B  MoE    NVFP4         224K    qwen3_coder   NVIDIA Omni, reasoning (mm disabled)
#
#   ctx = verified boot+completion ceiling on a single 32 GB 5090.
#
# ─── Picking one at a glance ────────────────────────────────────────────────
#   Fastest decode, and vision too?         → bonsai2        (~141 t/s ternary 27B, llama.cpp)
#   Best overall quality (newest Qwen)?     → qwen38-27b     (Qwen3.8 dense, dynamic NVFP4)
#   Newest Qwen but faster (some quality)?  → qwen38-fast    (MTP spec decode, ~1.6x)
#   Newest Qwen with vision?                → qwen38-vision  (image input, 131K)
#   Best coding quality per token?          → qwen36-27b-awq (dense, 2x decode)
#   Fastest capable daily driver + vision?  → qwen36         (3B active MoE)
#   Coder tool-loop, predictable latency?   → qwen3-coder    (no thinking blocks)
#   Strong reasoning + perfect tool-use?    → cascade2       (Mamba2+attn MoE)
#   Gemma 4 text+tool+reasoning?            → gemma4         (MoE AWQ, mm disabled)
#   Gemma 4 dense + vision?                 → gemma4-vision  (dense NVFP4, 0.24+)
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
#   Listen only on one network (binds to this host's address in that subnet):
#     BIND_CIDR=10.200.0.0/24 ./run.sh gpt-oss
#   Persist any of these in a .env file (auto-sourced); see .env.example:
#     cp .env.example .env && $EDITOR .env
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

# Load local defaults from .env if present (simple KEY=value lines, optional
# `export` prefix and # comments). Real environment variables take precedence,
# so `BIND_CIDR=… ./run.sh` still overrides the file. See .env.example.
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    _line="${_line%$'\r'}"                          # tolerate CRLF
    [[ "$_line" =~ ^[[:space:]]*(#|$) ]] && continue # skip comments / blanks
    _line="${_line#export }"
    [[ "$_line" == *=* ]] || continue
    _key="${_line%%=*}"; _key="${_key//[[:space:]]/}"
    [[ "$_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!_key+x}" ]] && continue                # already set → env wins
    _val="${_line#*=}"
    _val="${_val%\"}"; _val="${_val#\"}"             # strip surrounding quotes
    _val="${_val%\'}"; _val="${_val#\'}"
    export "$_key=$_val"
  done < "$SCRIPT_DIR/.env"
  unset _line _key _val
fi

IMAGE="vllm/vllm-openai:latest"
# Second runtime (RUNTIME=llamacpp models only; everything else uses IMAGE).
# Bonsai's ternary weights are unloadable by vLLM AND by stock llama.cpp: 402 of
# its 851 tensors use ggml type id 142, which exists only in prism-ml's fork,
# and the checkpoint carries prism.hadamard.* metadata for a Walsh-Hadamard
# activation transform applied at runtime. A loader that ignores that metadata
# returns fluent-looking garbage instead of an error, so there is no safe "just
# try it on the stock engine" path. No image is published — ./build-llamacpp.sh
# builds it, and run.sh auto-builds on a miss, like download-model.sh for weights.
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-bonsai-llamacpp:prism}"
CONTAINER_NAME="vllm"
HOST_PORT="${HOST_PORT:-8080}"
CONTAINER_PORT=8000

# ─── Resilience ─────────────────────────────────────────────────────────────
# RESTART_POLICY governs what Docker does when the server process EXITS (crash,
# OOM, CUDA error) or the host reboots. Default `unless-stopped`: bring the
# model back after a crash and after a reboot, but stay down if you explicitly
# ./stop-vllm.sh it. Docker applies exponential backoff, so a genuinely broken
# config won't hammer the GPU. Set RESTART_POLICY=no while debugging a config
# that won't boot, or on-failure:N to cap retries.
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
# Container HEALTHCHECK: vLLM has no curl, so probe /health with python3. This
# only marks the container healthy/unhealthy in `docker ps` — Docker does NOT
# auto-restart on "unhealthy" (a hung-but-alive server never exits). Pair it
# with ./watchdog-vllm.sh to restart on hangs. Long start-period so slow model
# loads + FlashInfer autotune warmup don't get flagged mid-boot.
HEALTHCHECK_CMD="python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:${CONTAINER_PORT}/health', timeout=5)\""

# ─── Network binding ────────────────────────────────────────────────────────
# By default the published port listens on every interface (0.0.0.0 — reachable
# from any network that can route to this host). Two ways to lock it down:
#   • HOST_IP=<addr>     — bind the published port to exactly that address
#                          (e.g. HOST_IP=127.0.0.1 for localhost-only).
#   • BIND_CIDR=<subnet> — restrict vLLM to a single network: run.sh finds THIS
#                          host's IPv4 address inside the subnet and binds the
#                          port only there, so the API is reachable only from
#                          that network. E.g. BIND_CIDR=10.200.0.0/24 binds to
#                          the wgN/VPN address and nothing else.
# HOST_IP (if set) wins over BIND_CIDR. A socket binds one address, not a CIDR,
# so this restricts the *interface*, not the source IP — for strict per-source
# filtering add a firewall rule (see README). For an isolated subnet/VPN,
# binding to its interface is the practical lock.
if [[ -z "${HOST_IP:-}" && -n "${BIND_CIDR:-}" ]]; then
  HOST_IP="$(ip -o -4 addr show to "$BIND_CIDR" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1 || true)"
  if [[ -z "$HOST_IP" ]]; then
    echo "run.sh: no local IPv4 address found within BIND_CIDR=$BIND_CIDR" >&2
    echo "  this host's IPv4 interfaces:" >&2
    ip -o -4 addr show 2>/dev/null | awk '{print "    "$2"\t"$4}' >&2
    exit 1
  fi
  echo ">>> BIND_CIDR=$BIND_CIDR → binding published port to $HOST_IP only" >&2
fi
HOST_IP="${HOST_IP:-0.0.0.0}"

# `--served-model-name` accepts multiple aliases — any of these names routes
# to whatever model is actually loaded. Keeps clients working without
# reconfiguration when switching models, and lets clients that hardcode a
# placeholder model ID (e.g. "gpt-4", "llama") continue to resolve.
SERVED_ALIASES=(
  default
  # This file's model keys:
  bonsai2 qwen38-27b qwen38-fast qwen38-vision qwen36 qwen36-fast qwen36-27b-nvfp4 qwen36-27b-awq qwen36-27b-unsloth qwen3-coder cascade2 gemma4 gemma4-vision gpt-oss nemotron3
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
  "bonsai2|27B dense ternary PQ2_0 (prism-ml), 262K, vision — ⭐ FASTEST 27B: ~141 t/s, 6.8 GB weights [llama.cpp fork, not vLLM]"
  "qwen38-27b|27B dense NVFP4-dynamic (unsloth), 262K — ⭐ NEWEST: Qwen3.8 QUALITY flavor, mm off (needs vLLM>=0.28)"
  "qwen38-fast|27B dense NVFP4+MTP (sakamakismile), 262K — Qwen3.8 SPEED flavor: MTP spec decode, mm off (needs vLLM>=0.28)"
  "qwen38-vision|27B dense NVFP4 (Inferact), 131K — Qwen3.8 VISION flavor: image input (needs vLLM>=0.28)"
  "qwen36-27b-awq|27B dense AWQ-INT4 (cyankiwi), 262K — ⭐ PREFERRED 27B: best quality/token, 2x faster decode"
  "qwen36-27b-nvfp4|27B dense NVFP4 (sakamakismile), 262K — Blackwell-native FP4, 27B dense"
  "qwen36-27b-unsloth|27B dense NVFP4-dynamic (unsloth), 262K — higher-quality NVFP4, mm off (needs vLLM>=0.24)"
  "qwen36-fast|35B/3B MoE NVFP4-dynamic (unsloth), 262K — 35B-A3B, thinks heavily, mm off (needs vLLM>=0.24)"
  "cascade2|30B/3B MoE NVFP4 (chankhavu), 131K — ⭐ Mamba2+attn, perfect tool-use, LiveCodeBench 87.2%"
  "qwen36|35B/3B MoE NVFP4 (RedHatAI), 196K, vision — newest Qwen flagship MoE, fastest capable decode"
  "qwen3-coder|30B/3B MoE AWQ (cyankiwi), 221K — non-thinking coder specialist, ~277 t/s"
  "gemma4|26B/4B MoE AWQ (cyankiwi), 262K — text+tool, mm disabled (86.4% τ²-bench)"
  "gemma4-vision|31B dense+vision NVFP4 (necroyancer), 128K — vision+reasoning Gemma4, ~69 t/s (verified 0.25.1)"
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
    bonsai2)
      # prism-ml/Ternary-Bonsai-2-27B-gguf, PQ2_0 (6.8 GB) + mmproj Q8_0
      # (601 MB). RUNTIME=llamacpp — this is the ONE model here that vLLM
      # cannot serve, at any version. Qwen3.8-27B retrained so its language
      # weights carry a ternary (-1/0/+1) representation packed at 2.13
      # bits/weight into ggml type 142, a type private to prism-ml's fork;
      # inference also applies a Walsh-Hadamard transform to activations
      # (prism.hadamard.* in the GGUF header) and an inverse transform on the
      # embedding lookup. Stock llama.cpp REFUSES PQ2_0/PTQ1_0 outright but
      # loads a legacy Q2_0 silently and emits gibberish — never "just try it".
      #
      # Same Gated DeltaNet hybrid backbone as qwen38-27b (64 layers, full
      # attention every 4th, ~75% linear) so KV stays cheap at depth. At 3x
      # smaller weights than the NVFP4 27Bs, the whole 262K context fits with
      # room to spare: VERIFIED 25,408 MiB of 32,607 at -c 262144 WITH the
      # vision projector resident (7.2 GB headroom). No --gpu-memory-utilization
      # knob exists here: llama.cpp allocates the KV cache from -c, it does not
      # carve a fraction of the card.
      #
      # ctx 262K (full native), VERIFIED: needle test recalled two codes planted
      # at 25%/75% depth in a 249,736-token prompt (also at 131,013).
      # SPEED (measured, batch 1, streaming): decode 142.1 t/s on short prompts
      # — 5x qwen38-27b (28.5) and 3.2x qwen38-fast (44.7). Decode DOES decay
      # with depth, unlike the vLLM dense entries which stay flat: 134 t/s @8K,
      # 115 @40K, 81 @131K, 58.8 @250K (still 2x qwen38-27b at full context).
      # Prefill is the trade — roughly half of vLLM's: 3,765 t/s @8K, 3,287
      # @40K, 1,896 @131K, 1,202 @250K (vLLM does ~2,100 at 259K). Deep prompts
      # cost real wall-clock: a 250K prefill is ~3.5 min. Short-to-mid context
      # chat is where this model wins outright.
      #
      # --jinja is REQUIRED for OpenAI tool calling (verified: emits proper
      # tool_calls). Thinking is always on (reasoning_content streams); cap it
      # per request with --reasoning-budget N. Vision verified through the
      # projector. The loader suggests --image-min-tokens 1024 for Qwen-VL
      # grounding accuracy; left unset here (untested, costs prefill).
      # Sampling defaults are the publisher's for Bonsai 2 (temp 1.0).
      RUNTIME="llamacpp"
      SNAPSHOT_REPO="prism-ml/Ternary-Bonsai-2-27B-gguf"
      GGUF_FILE="Ternary-Bonsai-2-27B-PQ2_0.gguf"
      MMPROJ_FILE="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
      MODEL_ARGS=(
        -c 262144
        -ngl 99
        -fa on
        --jinja
        --temp 1.0
        --top-p 0.95
        --top-k 20
      )
      ;;
    qwen38-27b)
      # unsloth/Qwen3.8-27B-NVFP4 (~22 GB). Unsloth Dynamic v3.0 NVFP4 of
      # Qwen3.8-27B — the direct successor of the Qwen3.6 27B dense VL line
      # (released 2026-08-13, Apache 2.0). Same Gated DeltaNet hybrid layout:
      # 64 layers, full attention every 4th → only 16 layers grow KV
      # (4 KV heads x 256 head_dim → ~32 KB/token at fp8). Mixed-precision
      # dynamic quant (lm_head kept FP8) = quality-first; ~1.4 GB LIGHTER than
      # the 3.6 unsloth so full-native ctx has more headroom.
      # Vision disabled via --limit-mm-per-prompt: text-only focus, skips
      # encoder profiling, frees memory for KV (context priority).
      # compressed-tensors → auto-detected (omit --quantization).
      # REQUIRES vLLM >= 0.28: Qwen3_5ForConditionalGeneration + FP8 lm_head
      # (verified on 0.28.0; the 0.25.1 image predates the arch).
      # MTP head ships in the weights — enable speculative decoding with
      #   --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
      # only if you can spare the VRAM; it shrinks the KV pool below full ctx.
      # ctx 262K (full native), VERIFIED on 0.28.0: KV 8.45 GiB → pool 269,809
      # tokens (1.03x). Needle test at 259K prompt tokens: exact recall of two
      # planted codes, 122s wall. --enforce-eager is mandatory at full ctx:
      # CUDA graphs need ~2.5 GiB and only ~0.45 GiB is spare after KV.
      # SPEED (measured, batch 1, streaming, TTFT excluded): decode ~28.5 t/s
      # FLAT across thinking / /no_think / JSON output (bandwidth-bound dense
      # decode — mode changes nothing). Prefill scales with depth: ~6.3K t/s
      # @38K ctx, ~3.4K @139K, ~2.1K @259K (eager, quadratic attn). TTFT
      # ~0.09s on short prompts.
      SNAPSHOT_REPO="unsloth/Qwen3.8-27B-NVFP4"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.95
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --limit-mm-per-prompt '{"image":0,"video":0}'
        --enable-auto-tool-choice
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
      )
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    qwen38-fast)
      # sakamakismile/Qwen3.8-27B-MTP-NVFP4 (~20 GB). SPEED flavor of the
      # Qwen3.8 trio (see qwen38-27b for quality, qwen38-vision for vision).
      # llm-compressor NVFP4 with the model's native MTP draft head kept in
      # BF16 (it's in quantization_config.ignore — REQUIRED: if the head gets
      # treated as NVFP4 the draft silently degrades to 0% acceptance) →
      # vLLM speculative decoding via method "mtp". Standard W4A4 quant =
      # some quality loss vs the unsloth dynamic quant; that's the trade.
      # Same DeltaNet hybrid constraints as the family. Vision disabled.
      # compressed-tensors → auto-detected. REQUIRES vLLM >= 0.28.
      # ctx 262K (full native), VERIFIED on 0.28.0: KV pool 277,296 tokens
      # (1.06x) — the MTP head shares embed/lm_head with the target, so spec
      # decoding costs almost no VRAM and full ctx survives.
      # SPEED (measured): ~44.7 t/s decode with num_speculative_tokens 3
      # (accept ~45%; k=2 gave 41, plain decode on this family is ~28.5) —
      # 1.57x the quality flavor. Tool calls verified faithful under spec.
      # MEMORY BOUNDARIES (verified crashes 2026-08-31, then fixed):
      #   --max-num-batched-tokens MUST stay 4096 — 8192 (copied from
      #   gpt-oss's spec tuning) OOMed AFTER boot on the first ~8K prefill.
      #   util MUST stay 0.93 — 0.95 booted fine but OOMed on a ~95K-token
      #   prefill (MTP activation footprint grows with depth).
      # At 4096/0.93: KV pool 275,549 tokens (1.05x); prefills VERIFIED OK at
      # 6K, 95K and 200K prompt tokens with spec decode active, engine stable.
      # KNOWN ISSUE (observed 2026-09-05, vLLM 0.28.0): a request can LIVELOCK
      # the engine — /health stays 200 while the GPU burns 99%/460W for hours
      # at ~0 tokens/s, and with --max-num-seqs 1 everything queues behind it.
      # Suspected MTP spec-decode edge case. watchdog-vllm.sh detects this
      # (frozen /metrics token counters with running>0) and restarts.
      SNAPSHOT_REPO="sakamakismile/Qwen3.8-27B-MTP-NVFP4"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.93
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --limit-mm-per-prompt '{"image":0,"video":0}'
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
        --enable-auto-tool-choice
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
      )
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    qwen38-vision)
      # Inferact/Qwen3.8-27B-NVFP4 (~25 GB — BF16 vision tower + MTP tensors
      # make it the heaviest Qwen3.8 quant). VISION flavor: native image +
      # video understanding left ON. ModelOpt NVFP4 → --quantization modelopt.
      # Image cap 2 / video off bounds the encoder profiling memory while
      # keeping two-image prompts usable. REQUIRES vLLM >= 0.28.
      # ctx 131072 IS THE VERIFIED CEILING (0.28.0): 25 GB weights + encoder
      # leave 3.64 GiB KV at util 0.95 (est. max 112,896); 0.96 → 123,872;
      # util 0.97 (the allocator's limit — do not exceed) just fits 131,072
      # with pool 134,085 tokens (1.02x). VERIFIED: two-color image described
      # exactly (colors + positions), coherent text, tool parser active.
      SNAPSHOT_REPO="Inferact/Qwen3.8-27B-NVFP4"
      MODEL_ARGS=(
        --quantization modelopt
        --max-model-len 131072
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.97
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --limit-mm-per-prompt '{"image":2,"video":0}'
        --enable-auto-tool-choice
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
      )
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    qwen36-27b-unsloth)
      # unsloth/Qwen3.6-27B-NVFP4 (~23.4 GB). Unsloth "dynamic" NVFP4 of the
      # Qwen3.6-27B dense VL model — mixed-precision (keeps sensitive layers
      # above 4-bit for quality), so ~3.7 GB heavier than the sakamakismile
      # NVFP4. DeltaNet hybrid (only the full-attention layers grow KV).
      # Vision disabled via --limit-mm-per-prompt: text-only focus, skips the
      # encoder profiling, and frees memory for KV given the heavy weights.
      # compressed-tensors → auto-detected (omit --quantization).
      # REQUIRES vLLM >= 0.24: unsloth quantizes the lm_head (ships
      # lm_head.weight_scale); vLLM 0.22.1's Qwen3_5 loader rejects that
      # ("no parameter named lm_head.weight_scale"). Verified on 0.24.0.
      # ctx 262K (full native): KV pool ~274K at 1.05x. Decode ~25 t/s — NVFP4
      # dense trades speed for quality; the AWQ 27B is ~2x faster.
      SNAPSHOT_REPO="unsloth/Qwen3.6-27B-NVFP4"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.95
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --limit-mm-per-prompt '{"image":0,"video":0}'
        --enable-auto-tool-choice
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
      )
      EXTRA_ENV+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      ;;
    qwen36-fast)
      # unsloth/Qwen3.6-35B-A3B-NVFP4-Fast (~23.6 GB). Unsloth speed-tuned NVFP4
      # of the Qwen3.6-35B-A3B MoE (256 experts / 8 active) VL model, mixed-
      # precision. Same family as `qwen36` (RedHatAI) but ~1.4 GB lighter.
      # DeltaNet hybrid MoE. Vision disabled for text-only + memory.
      # compressed-tensors → auto-detected.
      # REQUIRES vLLM >= 0.24 (same quantized-lm_head reason as the 27B).
      # ctx 262K (full native): KV pool ~955K at 3.64x — DeltaNet MoE stores
      # almost no KV, so context is free. Decode ~27 t/s (mixed-precision
      # NVFP4). NOTE: thinks heavily — give tool loops a large max_tokens or the
      # reasoning eats the budget before the tool call is emitted.
      SNAPSHOT_REPO="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"
      MODEL_ARGS=(
        --max-model-len 262144
        --max-num-batched-tokens 4096
        --max-num-seqs 1
        --gpu-memory-utilization 0.95
        --kv-cache-dtype fp8
        --enforce-eager
        --no-enable-prefix-caching
        --limit-mm-per-prompt '{"image":0,"video":0}'
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
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
    gemma4-vision)
      # necroyancer/gemma-4-31B-it-NVFP4-turbo-vision (~20.4 GB). NVFP4 (modelopt)
      # of Gemma 4 31B dense WITH the vision tower (Gemma4ForConditionalGeneration).
      # WHY THIS EXISTS: the earlier LilaRest text-only NVFP4 build (removed)
      # BROKE on vLLM >= 0.24 — its quantized lm_head hits gemma4.py's
      # tie_weights() which the modelopt quant method leaves NotImplemented.
      # necroyancer keeps the lm_head UNQUANTIZED (it's in the quant `ignore`
      # list), so tie_weights works → this is the 0.24+-compatible Gemma 4.
      # Reasoning parser is ENABLED and WORKS on 0.25.1 (the SentencePiece
      # unpickle bug that forced us to disable it on 0.22.1 is fixed). Vision is
      # left ON (no --limit-mm-per-prompt) — the encoder does NOT OOM here.
      # VERIFIED on the 5090 (vLLM 0.25.1): boots, coherent, faithful tool-calling,
      # ~69 t/s decode (thinking off). ctx 128K: KV pool ~179K tokens at 131072
      # (1.36x) — could push toward native 262144 with more util, vision keeps it
      # a bit tighter. SPEED is capped ~69 t/s (dense 31B): CUDA graphs are on;
      # FlashInfer attn is force-fallback'd to TRITON_ATTN (gemma4 sliding-window
      # + fp8 KV); and EAGLE3 spec decoding is IMPRACTICAL here — the RedHatAI
      # gemma-4-31B eagle3 draft is ~4.5 GB, which crushes KV to ~10K ctx on 32 GB
      # (vs gpt-oss's 0.3 GB head). No spec draft that fits leaves room for ctx.
      SNAPSHOT_REPO="necroyancer/gemma-4-31B-it-NVFP4-turbo-vision"
      MODEL_ARGS=(
        --quantization modelopt
        --max-model-len 131072
        --kv-cache-dtype fp8
        --enable-auto-tool-choice
        --tool-call-parser gemma4
        --reasoning-parser gemma4
        # prefix-caching, chunked-prefill, trust-remote-code come from COMMON_ARGS
      )
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
  # Newest revision first. A repo can have several cached once its upstream
  # revision moves; -t picks the current one (plain `ls` sorts by hash, which
  # is arbitrary). Single-file entries use resolve_cached_file instead.
  local snap
  snap=$(ls -1t "$cache_root" 2>/dev/null | head -1)
  [[ -n "$snap" ]] || { echo "run.sh: no snapshot inside $cache_root" >&2; exit 1; }
  printf '%s' "$cache_root/$snap"
}

# llama-server speaks a different flag language than vLLM. The web UI's
# override panel (and muscle memory) emit vLLM spellings, so translate the ones
# with a real equivalent and drop the ones that have none, loudly. Anything
# unrecognised passes straight through, so native llama-server flags still work.
translate_vllm_args() {
  local -a out=()
  while (( $# )); do
    case "$1" in
      --max-model-len)            out+=(-c "$2"); shift 2 ;;
      --max-num-seqs)             out+=(-np "$2"); shift 2 ;;
      --max-num-batched-tokens)   out+=(-b "$2"); shift 2 ;;
      --kv-cache-dtype)
        case "$2" in
          fp8|fp8_e4m3|fp8_e5m2)  out+=(--cache-type-k q8_0 --cache-type-v q8_0) ;;
          auto)                   ;;
          *) echo "run.sh: --kv-cache-dtype=$2 has no llama.cpp equivalent — ignored" >&2 ;;
        esac
        shift 2 ;;
      --gpu-memory-utilization)
        echo "run.sh: --gpu-memory-utilization is vLLM-only (llama.cpp sizes KV from -c) — ignored" >&2
        shift 2 ;;
      *) out+=("$1"); shift ;;
    esac
  done
  # Guard the empty case: printf on an empty array still emits one blank line,
  # which llama-server rejects with `invalid argument:`.
  if (( ${#out[@]} )); then printf '%s\n' "${out[@]}"; fi
}

# Locate ONE file inside a repo's cache, downloading just that file if absent.
#
# For GGUF entries we fetch file-by-file rather than whole-repo: such a repo
# ships several mutually exclusive quants of the same model (Bonsai's F16 is
# 53.8 GB on its own), so resolve_snapshot's whole-repo download — right for
# vLLM, where every shard is needed — would pull ~60 GB to use 7.4 GB of it.
#
# The catch is that HuggingFace opens a NEW snapshots/<rev>/ directory whenever
# the repo revision changes, so two files fetched weeks apart legitimately live
# under different revisions and NO single snapshot holds both. Resolve each file
# to wherever it actually is (newest revision first) instead of assuming one
# complete snapshot exists.
resolve_cached_file() {
  local repo="$1" want="$2"
  local root="$SCRIPT_DIR/cache/models--${repo//\//--}/snapshots"
  local hit
  hit=$(ls -1dt "$root"/*/"$want" 2>/dev/null | head -1 || true)
  if [[ -z "$hit" ]]; then
    echo ">>> $want not in cache — downloading it from $repo..." >&2
    "$SCRIPT_DIR/download-model.sh" "$repo" "*$want" >&2
    hit=$(ls -1dt "$root"/*/"$want" 2>/dev/null | head -1 || true)
  fi
  [[ -n "$hit" ]] || {
    echo "run.sh: $want not found in $repo even after downloading" >&2
    echo "  try: make download REPO=$repo GLOB='*$want'" >&2
    exit 1; }
  printf '%s' "$hit"
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
RUNTIME="vllm"       # "llamacpp" for models vLLM cannot load (see LLAMACPP_IMAGE)
GGUF_FILE=""         # llamacpp: weights file inside the snapshot dir
MMPROJ_FILE=""       # llamacpp: optional vision projector next to it
select_model "$target"

# Always launch detached. Tolerate a leading `-d` for backward compat.
if [[ "${1:-}" == "-d" ]]; then
  shift
fi

# HF cache snapshot/ entries are symlinks into ../../blobs/<hash>, so we must
# bind-mount the whole cache/ tree (not just the snapshot dir).
cache_path_in_container() { printf '/root/.cache/huggingface/%s' "${1#$SCRIPT_DIR/cache/}"; }

if [[ "$RUNTIME" == "llamacpp" ]]; then
  # Per-file, because these two may sit under different revisions (see
  # resolve_cached_file). Each gets its own container path.
  GGUF_HOST="$(resolve_cached_file "$SNAPSHOT_REPO" "$GGUF_FILE")"
  GGUF_CONTAINER="$(cache_path_in_container "$GGUF_HOST")"
  if [[ -n "$MMPROJ_FILE" ]]; then
    MMPROJ_HOST="$(resolve_cached_file "$SNAPSHOT_REPO" "$MMPROJ_FILE")"
    MMPROJ_CONTAINER="$(cache_path_in_container "$MMPROJ_HOST")"
  fi
  SNAPSHOT_HOST="$(dirname "$GGUF_HOST")"   # display only
else
  SNAPSHOT_HOST="$(resolve_snapshot "$SNAPSHOT_REPO")"
fi
SNAPSHOT_CONTAINER="$(cache_path_in_container "$SNAPSHOT_HOST")"

# Speculative-decoding draft head (if the model sets one): ensure it's in the
# mounted cache so vLLM resolves it by repo id offline, like the main weights.
[[ -n "$SPECULATOR_REPO" ]] && resolve_snapshot "$SPECULATOR_REPO" >/dev/null

# llamacpp runtime: the fork image has no registry to pull from, so build it
# locally on a miss (same contract as resolve_snapshot for weights).
if [[ "$RUNTIME" == "llamacpp" ]]; then
  if ! docker image inspect "$LLAMACPP_IMAGE" >/dev/null 2>&1; then
    echo ">>> $LLAMACPP_IMAGE missing — building it (one-off, ~20 min)" >&2
    "$SCRIPT_DIR/build-llamacpp.sh" || {
      echo "run.sh: failed to build $LLAMACPP_IMAGE" >&2; exit 1; }
  fi
fi

# Stop any previous container first; only one binds the port.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

mkdir -p "$SCRIPT_DIR/logs"

RUN_ARGS=(
  --name "$CONTAINER_NAME"
  # Record which lineup key launched this container. vLLM never sees it; the
  # web UI (ui/) reads it via `docker inspect` to know what's running, since
  # /v1/models always reports the full SERVED_ALIASES list.
  --label "vllm.model-key=$target"
  --runtime nvidia
  --gpus all
  --ipc host
  -p "${HOST_IP}:${HOST_PORT}:${CONTAINER_PORT}"
  -v "$SCRIPT_DIR/cache:/root/.cache/huggingface"
  -v "$SCRIPT_DIR/logs:/logs"
  -e "HF_HUB_CACHE=/root/.cache/huggingface"
  --health-cmd "$HEALTHCHECK_CMD"
  --health-interval 30s
  --health-timeout 10s
  --health-retries 3
  --health-start-period 300s
  "${EXTRA_ENV[@]}"
  "${EXTRA_VOLS[@]}"
)

# Optional bearer token gating vLLM's own /v1 endpoints (vLLM reads
# VLLM_API_KEY; equivalent to --api-key but passed as env so the secret never
# shows up in `ps` output or launch logs). The web UI always sets this to its
# API token so the model port is never open tokenless, wherever it binds; set
# VLLM_API_KEY in .env to gate manual launches too. /health and /metrics stay
# unauthenticated (healthcheck + monitoring depend on that).
# llama-server reads the same secret from LLAMA_API_KEY (its --api-key flag,
# as env so it stays out of `ps`); /health and /metrics stay open there too.
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  if [[ "$RUNTIME" == "llamacpp" ]]; then
    RUN_ARGS+=(-e "LLAMA_API_KEY=$VLLM_API_KEY")
  else
    RUN_ARGS+=(-e "VLLM_API_KEY=$VLLM_API_KEY")
  fi
fi

# Always detached. Restart policy = resilience (survive crash + reboot); see
# RESTART_POLICY above. Set RESTART_POLICY=no to debug a config that won't boot.
RUN_ARGS+=(-d --restart "$RESTART_POLICY")

display_ip="$HOST_IP"
[[ "$display_ip" == "0.0.0.0" ]] && display_ip="localhost"

echo ">>> model       : $target"
echo ">>> runtime     : $RUNTIME"
echo ">>> snapshot    : $SNAPSHOT_HOST"
echo ">>> endpoint    : http://${display_ip}:${HOST_PORT}/v1"
echo ">>> served name : default (+ aliases)"

if [[ "$RUNTIME" == "llamacpp" ]]; then
  # llama-server's own infrastructure flags. --alias takes the SAME alias list
  # vLLM gets via --served-model-name (comma-separated here), so any client
  # model ID keeps resolving; --metrics is OFF by default upstream and
  # watchdog-vllm.sh needs the counters to spot a livelock.
  LLAMACPP_ARGS=(
    -m "$GGUF_CONTAINER"
    --host 0.0.0.0
    --port "$CONTAINER_PORT"
    --alias "$(IFS=,; echo "${SERVED_ALIASES[*]}")"
    --metrics
  )
  [[ -n "$MMPROJ_FILE" ]] && LLAMACPP_ARGS+=(--mmproj "$MMPROJ_CONTAINER")
  # COMMON_ARGS is vLLM-only and deliberately NOT passed here.
  mapfile -t USER_ARGS < <(translate_vllm_args "$@")
  echo ">>> cli args    : ${MODEL_ARGS[*]} ${USER_ARGS[*]}"
  docker run "${RUN_ARGS[@]}" "$LLAMACPP_IMAGE" \
    "${LLAMACPP_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${USER_ARGS[@]}"
else
  echo ">>> cli args    : ${MODEL_ARGS[*]} ${COMMON_ARGS[*]} $*"
  # Order matters: COMMON_ARGS first (shared defaults), MODEL_ARGS second
  # (per-model overrides), "$@" last (user overrides everything). vLLM's argparse
  # takes the last-wins value for repeated flags.
  docker run "${RUN_ARGS[@]}" "$IMAGE" \
    "$SNAPSHOT_CONTAINER" \
    "${COMMON_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "$@"
fi

echo ">>> restart     : $RESTART_POLICY (survives crash + reboot; healthcheck on /health)"
echo ">>> started detached — tail with ./logs-vllm.sh, stop with ./stop-vllm.sh"
echo ">>> resilience  : for hang recovery, schedule ./watchdog-vllm.sh (see README)"
