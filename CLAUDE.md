# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

One-click vLLM serving for the RTX 5090 (32 GB, Blackwell sm_120). `run.sh`
launches a curated, hand-tuned model in a Docker container exposing an
OpenAI-compatible API on `:8080`. `pi.models.json` is the model registry for the
[pi coding agent](https://github.com/earendil-works/pi) so it can talk to that
server.

There is no build, no test suite, no linter — this repo is bash + JSON config.
"Testing" means booting a model on the real GPU and hitting it with a request.

## Commands

`make` with no target runs `up.sh`, the one-command path from nothing to a
served model (preflight → `run-ui.sh` → `run.sh` → wait for healthy). It is
idempotent: already-running pieces are left alone. It also hands the model the
UI's bearer token, so a `make`-launched model port is gated exactly like a
UI-launched one.

Every command below also has a `Makefile` target (`make help` lists them; e.g.
`make run MODEL=<key> ARGS="…"`, `make status`, `make ui`). The Makefile is a
thin wrapper — the scripts stay the source of truth, so a new script or flag
means adding a matching target. README instructions are written in `make` form;
keep them that way when adding commands.

```bash
make                                 # DEFAULT: preflight → UI → model → wait → report
make up MODEL=<key>                  # same, with a specific model (default: bonsai2)
./preflight.sh                       # host checks only; halts with instructions
./update-vllm.sh                     # docker pull vllm/vllm-openai:latest
./run.sh                             # interactive picker
./run.sh <key> [extra vllm args]     # boot detached; downloads weights if missing
./run.sh --list                      # keys + descriptions
./run.sh -h                          # prints the leading comment block of run.sh
./logs-vllm.sh                       # docker logs -f --tail 200 vllm
./stop-vllm.sh                       # docker rm -f vllm
./test-chat.sh "prompt"              # one-shot chat completion against :8080
./test-all-models.sh [key ...]       # boot → health → completion → stop, per model
./bench-ctx.sh [key ...]             # context-ceiling sweep → bench-ctx-results.txt
./download-model.sh <user/repo>      # or --all for DEFAULT_REPOS
./watchdog-vllm.sh                   # one-shot hang recovery; schedule via cron/timer
docker ps                            # healthy/unhealthy state of the `vllm` container
./run-ui.sh                          # web UI + token-gated OpenAI proxy on :8090
./stop-ui.sh | ./logs-ui.sh          # manage the vllm-ui container
```

Single-model verification (the closest thing to "running one test"):

```bash
./run.sh <key> && ./logs-vllm.sh     # wait for "Application startup complete";
                                     #   note "GPU KV cache size: N tokens" — N must be
                                     #   >= --max-model-len or it won't serve that ctx
./test-chat.sh "Write a haiku"       # output must be coherent, not garbage
./stop-vllm.sh
```

`./test-all-models.sh <key>` does that whole cycle unattended for one key.
For a context-ceiling check, send a prompt near `--max-model-len` and confirm it
doesn't OOM *and* recalls content (needle-in-haystack), not just that it boots.

Validate JSON after editing: `python3 -c "import json;json.load(open('pi.models.json'))"`.

## Golden rule

**Every config value in this repo is empirically verified on a real 32 GB
5090.** `ctx` columns mean "booted + completion-tested at this `--max-model-len`,
did not OOM." Never invent a context size, util, or quant flag from a model
card alone — boot it and confirm. If you change a flag, re-test before
committing.

## Architecture

Everything lives in `run.sh` (~650 lines). The flow, top to bottom:

1. **`.env` sourcing** — simple `KEY=value` lines, real env vars win. Holds
   `HOST_IP` / `HOST_PORT` / `BIND_CIDR` / `HF_TOKEN` / `RESTART_POLICY`.
2. **Network binding** — `BIND_CIDR` resolves *this host's* IPv4 inside that
   subnet and publishes the port only there (interface lock, not source
   filtering). `HOST_IP` overrides it. Default `0.0.0.0`.
3. **`SERVED_ALIASES`** — every model key plus generic placeholders (`default`,
   `gpt-4`, `llama`, `claude`, …) passed to `--served-model-name`, so any client
   model ID resolves to whatever is currently loaded.
4. **`COMMON_ARGS`** — shared defaults (util 0.92, chunked prefill, prefix
   caching on, `--max-num-seqs 64`).
5. **`MODELS`** — the picker list; its order is the interactive numbering.
6. **`select_model()`** — the heart of the repo. One `case <key>)` branch per
   model sets `SNAPSHOT_REPO`, `MODEL_ARGS`, and optionally `EXTRA_ENV`,
   `EXTRA_VOLS`, `SPECULATOR_REPO` (an EAGLE3 draft head, resolved into cache
   like the main weights — see `gpt-oss`). Each branch carries a dense comment
   encoding the VRAM math and OOM boundaries.
7. **`resolve_snapshot()`** — maps `user/repo` → `cache/models--user--repo/snapshots/<rev>/`,
   auto-invoking `download-model.sh` on a miss. The whole `cache/` tree is
   bind-mounted (snapshots symlink into `blobs/`, so mounting one snapshot dir
   breaks). `RUNTIME=llamacpp` entries use **`resolve_cached_file()`** instead:
   it fetches one named file at a time (a GGUF repo holds several mutually
   exclusive quants — Bonsai's F16 alone is 53.8 GB) and resolves each to
   whichever `snapshots/<rev>/` actually holds it, newest first. Those two files
   genuinely can land under different revisions, since HuggingFace opens a new
   snapshot dir whenever the repo revision moves, so each gets its own container
   path rather than being joined to one assumed-complete snapshot.
8. **`docker run`** — detached, `--restart unless-stopped`, `/health` healthcheck
   with a 300s start period. Arg order is `COMMON_ARGS` → `MODEL_ARGS` → `"$@"`;
   vLLM's argparse is **last-wins**, so your CLI args override everything.

### Two runtimes

Almost everything here is vLLM. One model (`bonsai2`) sets `RUNTIME="llamacpp"`
in its case block and launches `LLAMACPP_IMAGE` — prism-ml's llama.cpp fork,
built locally by `./build-llamacpp.sh` — because vLLM *cannot* load its weights
at any version: 402 of 851 tensors use ggml type id 142 (upstream ggml defines
0–41) and the GGUF carries `prism.hadamard.*` metadata for a runtime activation
transform. A loader that ignores it emits fluent garbage instead of failing, so
never "just try it" on the stock engine to check.

The branch is deliberately narrow — everything the rest of the stack depends on
is preserved rather than special-cased downstream:

- Same container name, same `CONTAINER_PORT`, same `vllm.model-key` label, same
  `/health` healthcheck command (the fork's image ships `python3` and `curl`).
- `SERVED_ALIASES` is passed as llama-server's comma-separated `--alias`, so
  alias resolution behaves the same (it accepts any `model` value anyway).
- The bearer token goes in as `LLAMA_API_KEY` instead of `VLLM_API_KEY`, so the
  port is never tokenless either way. **Caveat:** llama-server puts `/metrics`
  behind that key (vLLM leaves it open), which is why `watchdog-vllm.sh`
  authenticates its scrape — without that, a livelock reads as an idle server.
- `COMMON_ARGS` is vLLM-only and is **not** passed to llama-server.
- `translate_vllm_args()` maps the UI override panel's vLLM spellings to
  llama-server ones (`--max-model-len`→`-c`, `--max-num-seqs`→`-np`,
  `--max-num-batched-tokens`→`-b`, fp8 KV→`q8_0`), drops
  `--gpu-memory-utilization` (llama.cpp sizes KV from `-c`) with a warning, and
  passes anything unrecognised straight through.

`watchdog-vllm.sh` understands both metric namespaces (`vllm:*` and
`llamacpp:*`). The UI is runtime-aware too: `parse_run_sh()` returns `runtime`
per key, `fold_flags(..., short=True)` handles single-dash flags, and `-c` is
surfaced as `--max-model-len` so the chat context meter reads one key.

Resilience is layered: Docker's restart policy covers *exits* (crash, OOM,
reboot); the healthcheck only labels a hung-but-alive server `unhealthy`;
`watchdog-vllm.sh` is what actually restarts on a hang, and must be scheduled
externally (cron / systemd timer) — it is one-shot by design, do not loop it.

### Web UI (`ui/`)

FastAPI app + vanilla-JS frontend, run by `run-ui.sh` as container `vllm-ui`
(`--network host`, docker socket + repo mounted at its identical host path,
non-root). Password gate = `UI_PASSWORD` env; the OpenAI API is re-served at
`:8090/v1` behind a single bearer token (state in `ui/data/state.json`,
gitignored). The Monitor tab's history charts are fed by an in-memory 5s
sampler in `app.py` (`HISTORY` ring, last hour — cleared on UI restart by
design). Chat web browsing (`ui/browse.py`): `/api/chat` with
`"browsing": true` runs a server-side tool loop — `web_search`/`web_fetch`
tools injected, executed as per-call `docker run --rm h4ckf0r0day/obscura`,
activity + debug streamed as `{"browsing": …}` SSE events. Obscura's
private-network SSRF protection is intentionally left on (never set
`OBSCURA_ALLOW_PRIVATE_NETWORK`); tool calling requires the model's
`--tool-call-parser`; completed turns replay flattened (assistant text only,
no historical tool messages). Python execution (`ui/pyexec.py` +
`ui/sandbox/`): the same loop's `run_python` tool, one throwaway
`vllm-pysandbox` container per call — `--network none`, read-only rootfs,
resource caps, 30s timeout; libraries are baked into the image because pip
at runtime is impossible by design. Figures auto-save (runner.py) and stream
to the frontend as data URLs; the model only gets text. The image is code-free (deps + docker CLI + iproute2 only) — the app runs
from the mounted `ui/`, so it can't drift from `run.sh` across rebuilds; a
`docker restart vllm-ui` picks up code changes, no rebuild needed. The Chat
tab talks to the running model through session-gated `/api/chat` (the browser
never sees the bearer token); conversations live in browser localStorage.

Key contracts to preserve when editing:

- The UI **executes `./run.sh <key> [overrides]`**. It never re-implements
  launch flags. The serving bind is user-configurable (API access panel →
  "Model serving bind", persisted as `serve_bind` in state): unset = pass no
  `HOST_IP`/`HOST_PORT` so run.sh applies `.env`; an explicit choice is passed
  as real env vars (which win over `.env`). Every UI launch also sets
  `VLLM_API_KEY` to the UI's bearer token — run.sh forwards it into the
  container, so the model's direct /v1 port is never tokenless wherever it
  binds (/health + /metrics stay open; the proxy authenticates upstream with
  the key from `docker inspect` Env, so token renewal never breaks it). The
  UI image ships `iproute2` because run.sh's `BIND_CIDR` path shells out to
  `ip`.
- `ui/vllm_mgr.py` **structurally parses `run.sh`**: `select_model()`'s
  `key)` labels, `SNAPSHOT_REPO="…"`, `MODEL_ARGS=( … )`, plus `COMMON_ARGS`
  and `./run.sh --list`. Keep that layout, or fix the parser in the same
  change. It also merges metadata from `pi.models.json` by key.
- `run.sh` stamps `--label "vllm.model-key=$target"` on the container — the
  UI's primary way to know what's loaded (`/v1/models` can't tell: it always
  returns the full `SERVED_ALIASES` list). Snapshot-path fallback covers
  pre-label containers.
- The proxy derives the upstream address from `docker inspect vllm`
  (PortBindings, `0.0.0.0`→`127.0.0.1`), so it also serves models launched
  manually, wherever they bound.

## Adding a new model

A model is fully added only when **all five** of these are updated. Missing any
one leaves the repo inconsistent.

### 1. `run.sh` — `select_model()` case block

Add a new `<key>)` branch. Set `SNAPSHOT_REPO` to the HuggingFace repo and
`MODEL_ARGS` to the launch flags. (A GGUF model that vLLM cannot load also sets
`RUNTIME="llamacpp"` plus `GGUF_FILE` / `MMPROJ_FILE` — see `bonsai2`, and note
its `MODEL_ARGS` are llama-server flags, not vLLM ones.) Write a comment block above it explaining:
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

### Helper scripts carry their own model lists — and they drift

Three helpers duplicate the lineup (in sync as of the `qwen38-27b` addition).
Update them alongside the five above, or fix the drift when you touch them:

- `download-model.sh` → `DEFAULT_REPOS` (drives `--all`; a missing entry means
  `--all` silently skips that model, though `run.sh` still auto-downloads it).
- `test-all-models.sh` → `ALL_MODELS`.
- `bench-ctx.sh` → `ALL_MODELS` (entries are `key|target_ctx|notes`, and
  `target_ctx` must match `--max-model-len` in `run.sh`).

## Conventions

- Commit only when asked. Match the existing author (`Pau <pau@dabax.net>`).
- Keep comments in `run.sh` dense and specific — they encode hard-won OOM
  boundaries; do not trim them to "clean up."
- The container tracks `vllm/vllm-openai:latest`. Flags can break across vLLM
  releases (the `unsloth` NVFP4-dynamic entries need ≥0.24; a quantized
  `lm_head` broke the old Gemma 4 text model on ≥0.24). Note the verified vLLM
  version in the comment when a model is version-sensitive.
