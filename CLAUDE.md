# CLAUDE.md

One-click vLLM serving for the RTX 5090 (32 GB, Blackwell sm_120). `run.sh`
launches a curated, hand-tuned model in a Docker container exposing an
OpenAI-compatible API on `:8080`. `pi.models.json` is the model registry for the
[pi coding agent](https://github.com/earendil-works/pi) so it can talk to that
server.

## Repo layout

- `run.sh` — the launcher. Per-model launch flags live in `select_model()`.
- `pi.models.json` — pi coding agent model registry (`~/.pi/agent/models.json`
  format). **Must be kept in sync with the models in `run.sh`.**
- `README.md` — user-facing lineup table + rationale.
- `templates/` — chat templates bind-mounted into the container (Gemma 4).
- `download-model.sh`, `stop-vllm.sh`, `update-vllm.sh`, `logs-vllm.sh`,
  `test-chat.sh`, `test-all-models.sh`, `bench-ctx.sh` — helpers.

## Golden rule

**Every config value in this repo is empirically verified on a real 32 GB
5090.** `ctx` columns mean "booted + completion-tested at this `--max-model-len`,
did not OOM." Never invent a context size, util, or quant flag from a model
card alone — boot it and confirm. If you change a flag, re-test before
committing.

## Adding a new model

A model is fully added only when **all five** of these are updated. Missing any
one leaves the repo inconsistent.

### 1. `run.sh` — `select_model()` case block

Add a new `<key>)` branch. Set `SNAPSHOT_REPO` to the HuggingFace repo and
`MODEL_ARGS` to the launch flags. Write a comment block above it explaining:
weights size, architecture, why each non-obvious flag is set, and the verified
`ctx` ceiling (with what OOMs above it).

Common knobs (defaults in `COMMON_ARGS`; per-model `MODEL_ARGS` override them;
your CLI args override everything — last-wins):
- `--max-model-len` — the verified context ceiling.
- `--max-num-seqs 1` — single-user serving; frees the whole KV pool for one
  sequence. Set this for any model where you want maximum context.
- `--gpu-memory-utilization` — higher = more KV, but leave headroom for the
  warmup forward pass and autotuner (going too high OOMs *after* KV allocation).
- `--kv-cache-dtype fp8` — halves KV on Blackwell; use unless a model misbehaves.
- `--quantization` — set explicitly for `modelopt_fp4` (NVFP4) when vLLM does
  not auto-detect it; omit for AWQ/compressed-tensors that self-describe.
- `--tool-call-parser` / `--reasoning-parser` — must match the model family.
- `--enforce-eager`, `--no-enable-prefix-caching` — needed by some hybrid/
  recurrent architectures (DeltaNet, Mamba2) and to save memory.
- `--limit-mm-per-prompt` — disable image/video/audio to skip encoder profiling
  that OOMs on text-only use.
- `EXTRA_VOLS` / `EXTRA_ENV` — bind a chat template, or set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fight fragmentation.

### 2. `run.sh` — `SERVED_ALIASES`

Add the new key so vLLM serves the loaded model under that name (clients
addressing it by key resolve correctly). Aliases for the same model may share a
case branch (e.g. `gemma4-text|gemma4-coder`).

### 3. `run.sh` — `MODELS` array + header comment

- Add a `"<key>|<desc>"` entry to the `MODELS` array (the interactive picker).
- Add a row to the `─── Model lineup ───` table in the leading comment, and a
  line to the `─── Picking one at a glance ───` guide if it fills a new role.

### 4. `README.md`

Add a row to the **Model lineup** table (`key | params | quant | ctx | vision |
role`) and update the disk-footprint total if relevant.

### 5. `pi.models.json` — REQUIRED, do not skip

Add a model object under `providers.vllm.models`. This is what makes the model
usable from the pi coding agent. **Keep it in lockstep with `run.sh`** — same
`id`, same context size.

```json
{
  "id": "<key>",                      // MUST equal the run.sh key / a served alias
  "name": "<Model> <quant> (vLLM) — <ctx>, <notes>",
  "reasoning": true,                  // true only if a --reasoning-parser is ACTIVE
  "input": ["text"],                  // add "image" only if vision is served (mm not disabled)
  "contextWindow": 262144,            // MUST equal --max-model-len in run.sh
  "maxTokens": 32768,                 // output cap; 32768 is the repo convention
  "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
  "compat": { "thinkingFormat": "qwen-chat-template" }  // Qwen thinking only; omit otherwise
}
```

Field rules:
- `contextWindow` **must** equal the model's `--max-model-len` in `run.sh`. If
  you change the ceiling in one file, change it in the other.
- `reasoning`: `true` only when a `--reasoning-parser` is actually enabled and
  emitting `reasoning_content`. If the parser is disabled (e.g. the Gemma 4
  tokenizer bug) or the model is a non-thinking Instruct, set `false`.
- `input`: `["text","image"]` only when vision is the intended, tested path.
  If `run.sh` sets `--limit-mm-per-prompt` to suppress the vision encoder (even
  `image:1` as an OOM workaround, as `gemma4` does), treat it as `["text"]`.
- `compat.thinkingFormat: "qwen-chat-template"` for the Qwen3.x family (they
  need `chat_template_kwargs.enable_thinking`). Other families omit it; the
  provider-level `supportsDeveloperRole:false` / `supportsReasoningEffort:false`
  already cover vLLM's quirks.

Validate after editing: `python3 -c "import json;json.load(open('pi.models.json'))"`.

## Verifying a model

```bash
./run.sh <key>                 # boots detached, downloads weights if missing
./logs-vllm.sh                 # watch for "Application startup complete"; note the
                               #   "GPU KV cache size: N tokens" line — N must be
                               #   >= your --max-model-len or it won't boot
./test-chat.sh "Write a haiku" # smoke test; check output is coherent, not garbage
./stop-vllm.sh
```

For a context-ceiling check, send a prompt near `--max-model-len` and confirm it
doesn't OOM and recalls content (needle-in-haystack), not just that it boots.

## Conventions

- Commit only when asked. Match the existing author (`Pau <pau@dabax.net>`).
- Keep comments in `run.sh` dense and specific — they encode hard-won OOM
  boundaries; do not trim them to "clean up."
