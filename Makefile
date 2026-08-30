# Makefile — one entry point for every repository command.
# All targets are thin wrappers around the scripts (the scripts stay the source
# of truth and remain directly runnable). `make` / `make help` lists everything.
#
# Variables:
#   MODEL=<key>     model for run/test-all/bench     (make run MODEL=gpt-oss)
#   ARGS="..."      extra vllm serve args for run    (make run MODEL=qwen36 ARGS="--max-model-len 65536")
#   PROMPT="..."    prompt for test-chat
#   REPO=user/repo  HuggingFace repo for download

.DEFAULT_GOAL := help
.PHONY: help update run list stop logs test-chat test-all bench download \
        download-all watchdog ui ui-stop ui-logs status validate

help: ## Show this help
	@echo "rtx5090-vllm — available targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables: MODEL=<key>  ARGS=\"...\"  PROMPT=\"...\"  REPO=user/repo"

# ─── serving ────────────────────────────────────────────────────────────────

update: ## Pull the latest vllm/vllm-openai image
	./update-vllm.sh

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

download: ## Download one model (REPO=user/repo)
	@test -n "$(REPO)" || { echo "usage: make download REPO=user/repo"; exit 1; }
	./download-model.sh $(REPO)

download-all: ## Download every model in DEFAULT_REPOS
	./download-model.sh --all

# ─── web UI ─────────────────────────────────────────────────────────────────

ui: ## Build + start the web UI / token-gated proxy (needs UI_PASSWORD in .env)
	./run-ui.sh

ui-stop: ## Stop the vllm-ui container
	./stop-ui.sh

ui-logs: ## Tail the vllm-ui container logs
	./logs-ui.sh
