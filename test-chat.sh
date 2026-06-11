#!/usr/bin/env bash
# Send a one-shot chat completion to the running vLLM server and print
# the assistant reply. Use to verify a model is alive + coherent.
#
# Usage:
#   ./test-chat.sh                      # simple factorial prompt
#   ./test-chat.sh "your prompt here"
#
# Env:
#   ENDPOINT   base URL of the server (default: http://localhost:8080/v1)
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8080/v1}"
PROMPT="${1:-Write a Python function that returns the factorial of n. One function, no imports.}"
RESP=$(curl -sS --max-time 120 "$ENDPOINT/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$PROMPT" '{model: "default", max_tokens: 2000, temperature: 0.2, messages: [{role: "user", content: $p}]}')")

if command -v jq >/dev/null 2>&1; then
    echo "$RESP" | jq -r '.choices[0].message.content // .error.message // .'
else
    echo "$RESP"
fi
