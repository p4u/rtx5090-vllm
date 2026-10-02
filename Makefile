# Makefile — one entry point for every repository command.
# All targets are thin wrappers around the scripts (the scripts stay the source
# of truth and remain directly runnable). `make` brings the whole stack up;
# `make help` lists everything.
#
# Variables:
#   MODEL=<key>     model for up/run/test-all/bench  (make up MODEL=qwen38-27b)
#   ARGS="..."      extra vllm serve args for run    (make run MODEL=qwen36 ARGS="--max-model-len 65536")
#   PROMPT="..."    prompt for test-chat
#   REPO=user/repo  HuggingFace repo for download
#   GLOB="*.gguf"   restrict a download to matching files

.DEFAULT_GOAL := up
.PHONY: help up preflight update run list stop logs test-chat test-all bench \
        download download-all watchdog ui ui-stop ui-logs status validate \
        build-llamacpp

help: ## Show this help
	@echo "rtx5090-vllm — available targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables: MODEL=<key>  ARGS=\"...\"  PROMPT=\"...\"  REPO=user/repo  GLOB=\"*.gguf\""

# ─── everything at once ─────────────────────────────────────────────────────

up: ## ⭐ Default: bring the whole stack up — UI + a served model (MODEL=<key>)
	./up.sh

preflight: ## Check this host has everything needed (run by `up`)
	./preflight.sh

# ─── serving ────────────────────────────────────────────────────────────────

update: ## Pull the latest vllm/vllm-openai image
	./update-vllm.sh

build-llamacpp: ## Build the llama.cpp fork image (RUNTIME=llamacpp models, e.g. bonsai2)
	./build-llamacpp.sh

run: ## Boot a model (MODEL=<key>, else interactive picker; ARGS= extra vllm args)
	./run.sh $(MODEL) $(ARGS)

list: ## List model keys + descriptions
	./run.sh --list

stop: ## Stop and remove the vllm container
	./stop-vllm.sh

logs: ## Tail the vllm container logs
	./logs-vllm.sh

status: ## Show vllm + vllm-ui container status
	@docker ps -a --filter name=vllm --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Label "vllm.model-key"}}'

watchdog: ## One-shot hang recovery (schedule via cron/systemd, don't loop)
	./watchdog-vllm.sh

# ─── testing ────────────────────────────────────────────────────────────────

test-chat: ## One-shot chat completion against :8080 (PROMPT="...")
	./test-chat.sh $(if $(PROMPT),"$(PROMPT)")

test-all: ## Boot → health → completion → stop for every model (or MODEL=<key>)
	./test-all-models.sh $(MODEL)

bench: ## Context-ceiling sweep → bench-ctx-results.txt (or MODEL=<key>)
	./bench-ctx.sh $(MODEL)

validate: ## Validate pi.models.json syntax
	python3 -c "import json;json.load(open('pi.models.json'))" && echo "pi.models.json OK"

# ─── weights ────────────────────────────────────────────────────────────────

download: ## Download one model (REPO=user/repo, optional GLOB="*.gguf")
	@test -n "$(REPO)" || { echo "usage: make download REPO=user/repo [GLOB='*.gguf']"; exit 1; }
	./download-model.sh $(REPO) $(if $(GLOB),"$(GLOB)")

download-all: ## Download every model in DEFAULT_REPOS
	./download-model.sh --all

# ─── web UI ─────────────────────────────────────────────────────────────────

ui: ## Build + start the web UI / token-gated proxy (needs UI_PASSWORD in .env)
	./run-ui.sh

ui-stop: ## Stop the vllm-ui container
	./stop-ui.sh

ui-logs: ## Tail the vllm-ui container logs
	./logs-ui.sh
