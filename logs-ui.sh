#!/usr/bin/env bash
# Tail the vllm-ui container logs.
exec docker logs -f --tail 200 vllm-ui
