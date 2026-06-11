# rtx5090-vllm

**One-click vLLM serving for the RTX 5090 (32 GB, Blackwell sm_120).**

A curated set of large language models that actually fit on a single RTX 5090,
each with a hand-tuned launch config (quant, context length, KV precision,
tool-call parser) verified to boot and serve on a 32 GB card. Pick a model and
go — weights download automatically on first run.

```bash
./run.sh                 # interactive picker → downloads if needed → serves on :8080
./run.sh qwen36-27b-awq  # or name a model directly
```

The server exposes an **OpenAI-compatible HTTP API** at
`http://<host>:8080/v1`, so it drops straight into any OpenAI client, agent
framework, or coding tool.

---

## Why this exists

The 5090's 32 GB of VRAM and **native NVFP4/MXFP4 tensor cores** put it in an
awkward spot: big enough for serious 27–35B models, but only with the right
quant and context math. Get a flag wrong and you OOM on boot, silently fall
back to a slow kernel, or get garbage tool calls from a mismatched parser.

This repo encodes the working configs so you don't have to rediscover them.
Every model in the lineup has been booted and completion-tested on a real
32 GB 5090.

---

## Requirements

- **RTX 5090** (32 GB). Other 32 GB Blackwell cards likely work; smaller cards
  will OOM on most models — drop `--max-model-len` and bump quant aggressiveness.
- **NVIDIA driver** with CUDA 12.8+ (Blackwell sm_120 support).
- **Docker** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`--runtime nvidia --gpus all` must work).
- **`jq`** and **`curl`** for the test scripts.
- **`hf` CLI** (`pip install "huggingface_hub[hf_xet]"`) for fast downloads —
  optional, a Docker-based fallback is built in.
- Disk: ~20 GB per model. The full lineup is ~180 GB.

No local Python/PyTorch/CUDA install needed — vLLM runs entirely inside the
`vllm/vllm-openai:latest` container.

---

## Quickstart

```bash
git clone https://github.com/<you>/rtx5090-vllm.git
cd rtx5090-vllm

# 1. Pull the vLLM image (once).
./update-vllm.sh

# 2. Launch a model. Weights download to ./cache automatically if missing.
./run.sh qwen36-27b-awq

# 3. Wait for boot (30–120s), then smoke-test it.
./logs-vllm.sh              # watch until "Application startup complete"
./test-chat.sh "Write a haiku about GPUs."

# 4. Use it from any OpenAI client.
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'

# 5. Stop / switch.
./stop-vllm.sh
./run.sh gpt-oss
```

> **Binding:** by default the server binds `0.0.0.0:8080` (reachable on your
> LAN). For localhost-only, run `HOST_IP=127.0.0.1 ./run.sh <model>`.

---

## Model lineup

Nine models, each filling a specific role. Run `./run.sh --help` for the full
per-model rationale, or `./run.sh` for the interactive picker.

| key                | params        | quant      | ctx (5090) | vision | role |
|--------------------|---------------|------------|------------|--------|------|
| `qwen36-27b-awq`   | 27B dense     | AWQ INT4   | 262K       | —      | ⭐ Best coding quality/token, ~2× decode vs NVFP4 |
| `qwen36-27b-nvfp4` | 27B dense     | NVFP4      | 262K       | —      | Same model, Blackwell-native FP4 path |
| `cascade2`         | 30B/3B MoE    | NVFP4      | 131K       | —      | ⭐ Mamba2+attn, perfect tool-use, LiveCodeBench 87.2 |
| `qwen36`           | 35B/3B MoE    | NVFP4      | 196K       | ✓      | Newest Qwen flagship, fastest capable decode |
| `qwen3-coder`      | 30B/3B MoE    | AWQ INT4   | 221K       | —      | Non-thinking coder specialist, ~277 t/s |
| `gemma4`           | 26B/4B MoE    | AWQ INT4   | 262K       | —      | Google text+tool, 86.4% τ²-bench (mm disabled) |
| `gemma4-text`      | 27B dense     | NVFP4      | 32K        | —      | Text-only Gemma 4, no multimodal overhead |
| `gpt-oss`          | 21B/3.6B MoE  | MXFP4      | 131K       | —      | OpenAI open weights, `Reasoning: low/med/high` |
| `nemotron3`        | 31B/3B MoE    | NVFP4      | 224K       | —      | NVIDIA Omni reasoning MoE (mm disabled) |

`ctx` = verified boot + completion ceiling on a single 32 GB card with fp8 KV.
All values confirmed on vLLM 0.22.1.

### Picking one at a glance

- **Best coding quality per token** → `qwen36-27b-awq` (dense, 2× decode)
- **Fastest capable daily driver + vision** → `qwen36` (3B-active MoE)
- **Tool-loop with predictable latency** → `qwen3-coder` (no thinking blocks)
- **Strong reasoning + perfect tool-use** → `cascade2`
- **Google Gemma 4 text+tool** → `gemma4`
- **OpenAI weights with a reasoning dial** → `gpt-oss`

---

## How it works

- vLLM runs in the `vllm/vllm-openai:latest` Docker container, one model at a
  time, bound to `:8080`.
- Weights live in `./cache/` using HuggingFace's
  `models--<user>--<repo>/snapshots/<rev>/` layout and are bind-mounted into
  the container (no network at serve time).
- `--served-model-name` registers a long list of **aliases** (`default`, every
  model key, plus generic placeholders like `gpt-4`, `llama`, `model`). Any of
  those route to whatever model is currently loaded, so clients that hardcode a
  model ID work without reconfiguration.
- Per-model launch flags live in `select_model()` in `run.sh`. They are
  measured fits for 32 GB — read the inline comments before changing them.

```
run.sh                  # main launcher: ./run.sh (picker) | ./run.sh <model> [args]
download-model.sh       # ./download-model.sh <user/repo> | --all
stop-vllm.sh            # docker rm -f vllm
update-vllm.sh          # docker pull vllm/vllm-openai:latest
logs-vllm.sh            # docker logs -f --tail 200 vllm
test-chat.sh            # one-shot chat completion against the server
test-all-models.sh      # boot every model, health-check, completion, summarize
bench-ctx.sh            # context-ceiling sweep across the lineup
templates/              # chat templates mounted into the container (Gemma 4)
cache/                  # downloaded weights (gitignored)
```

### Common overrides

```bash
# Push context above the per-model default:
./run.sh qwen3-coder --max-model-len 262144

# Free more VRAM for KV:
./run.sh qwen36 --gpu-memory-utilization 0.97

# Different bind address / port:
HOST_IP=127.0.0.1 HOST_PORT=9090 ./run.sh gemma4

# Any extra vllm serve args are forwarded verbatim:
./run.sh qwen3-coder --max-num-seqs 64
```

Arg precedence is last-wins: shared defaults → per-model flags → your CLI args.

### Download ahead of time

```bash
./download-model.sh --all                                   # every model in the lineup
./download-model.sh cyankiwi/Qwen3.6-27B-AWQ-INT4           # a specific repo
```

Export `HF_TOKEN=hf_...` to avoid unauthenticated HuggingFace rate limits.

---

## Notes on the RTX 5090 (Blackwell)

The 32 GB of VRAM is the binding constraint: weights + KV cache + activations +
CUDA-graph buffers must all fit. After a ~20 GB weight load you have ~10 GB for
everything else.

**Quant format ranking for throughput × quality on sm_120:**

1. **NVFP4** — native FP4 tensor-core path, no dequant, best precision at 4-bit.
2. **MXFP4** — same FP4 path, slightly different scale layout (gpt-oss MoE).
3. **AWQ INT4** — mature, dequants to FP16/BF16; a step slower than NVFP4 but
   often the most *robust* and, in practice, faster to decode on some kernels.
4. **GPTQ INT4** — similar to AWQ, marginally slower.
5. **FP8** — usually too big for 32 GB above ~13B params.
6. **GGUF** — *not supported by vLLM*; use [llama.cpp](https://github.com/ggml-org/llama.cpp).

### Context-maximization ladder

Want max context without OOM? Apply in order:

1. Read the model's `config.json` → `max_position_embeddings` is the hard
   native ceiling (above it vLLM errors, it doesn't OOM).
2. Try native ctx + `--kv-cache-dtype fp8`. Check the boot log for
   `Available KV cache memory: N GiB`.
3. OOM by ≤1 GB → bump `--gpu-memory-utilization` from 0.92 toward 0.97
   (~300 MB per 0.01 step; don't exceed 0.97, the allocator needs headroom).
4. OOM by 1–3 GB → add `--enforce-eager` (frees ~2.5 GB of CUDA-graph memory;
   costs ~6% decode and hurts TTFT on big prefills). Optionally lower
   `--max-num-seqs`.
5. OOM by more → drop `--max-model-len` to the value vLLM prints in the error.

### Tool-call parser cheat sheet

A mismatched `--tool-call-parser` is the most common "it kind of works but tool
calls are broken" failure. The call leaks into `content` as raw text instead of
populating `tool_calls`. Common parsers:

| parser        | emitted format                                          | used by |
|---------------|---------------------------------------------------------|---------|
| `qwen3_coder` | `<tool_call>...<function=...>...</tool_call>` XML        | Qwen3-Coder, Qwen3.6 MoE, Nemotron |
| `qwen3_xml`   | Qwen3 XML variant                                       | Qwen3.6 27B dense |
| `gemma4`      | `<\|tool_call>call:NAME{args}<tool_call\|>`             | Gemma 4 family |
| `openai`      | OpenAI JSON / function_call                             | gpt-oss, OpenAI clones |

`--reasoning-parser` is separate; it splits `<think>` blocks into the response's
`reasoning` field (`qwen3`, `nemotron_v3`, `openai_gptoss`, …).

> vLLM ≥0.22.1 renamed the response field `reasoning_content` → `reasoning`.
> Clients reading the chain-of-thought should handle both.

---

## Gotchas

- **One container, one port.** Only one model serves at a time on `:8080`.
  `run.sh` stops any previous container before starting. Switching models means
  `./stop-vllm.sh` then `./run.sh <other>` (30–120s reload).
- **Root-owned cache after first run.** The container runs as root internally,
  so `cache/` and `logs/` end up root-owned, which breaks user-mode
  `hf download`. Fix once: `sudo chown -R "$USER:$USER" cache logs`.
- **DeltaNet hybrids need prefix caching off.** The Qwen3.6 family carries a
  recurrent state not reflected in KV blocks; prefix caching produces wrong
  outputs. Already handled per-model (`--no-enable-prefix-caching` +
  `--max-num-batched-tokens 4096`).
- **compressed-tensors vs awq_marlin.** Some AWQ quants ship as
  `compressed-tensors`. If vLLM complains the quantization method doesn't match,
  drop `--quantization` and let it auto-detect.
- **Mount the whole cache tree.** HF snapshots symlink into `blobs/`; bind-mount
  the entire `cache/` dir, not just one snapshot, or the symlinks break inside
  the container. `run.sh` already does this.

---

## Adding a model

1. Confirm on-disk weight size leaves room for KV (≤ ~20 GB is comfortable on
   32 GB; > 26 GB will OOM at any useful context).
2. Prefer NVFP4 → MXFP4 → AWQ on Blackwell. Avoid GGUF (use llama.cpp).
3. Verify the chat template matches a vLLM-registered tool-call parser **before**
   committing to a download.
4. Add a `case <key>)` block in `select_model()` (`run.sh`) with the launch
   flags and a comment explaining the VRAM/context math.
5. Add the key + description to the `MODELS` array and `SERVED_ALIASES`.
6. Add the repo to `DEFAULT_REPOS` in `download-model.sh`.
7. Verify with `./test-all-models.sh <key>`.

---

## License

MIT — see [LICENSE](LICENSE).

vLLM, the models, and their quantizations are distributed under their own
licenses; check each model's HuggingFace card before commercial use.
