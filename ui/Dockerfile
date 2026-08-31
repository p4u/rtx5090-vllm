# vllm-ui — model manager + token-gated OpenAI proxy for rtx5090-vllm.
#
# Deliberately code-free: the image carries only the Python deps and a docker
# CLI. run-ui.sh bind-mounts the repo at its identical host path and runs the
# app from the mounted ui/ dir, so the UI, run.sh, and pi.models.json can never
# drift apart across image rebuilds — and the UI's run.sh parser always sees
# the run.sh it will execute.
FROM python:3.12-slim

# Static docker CLI (no daemon) — talks to the host daemon via the mounted
# /var/run/docker.sock. This is how the UI starts/stops/inspects `vllm`.
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

# iproute2: run.sh's BIND_CIDR resolution shells out to `ip`, and the UI
# enumerates host networks (--network host makes them visible) for the
# serving-bind presets.
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The container runs as the host user (run-ui.sh passes --user), which has no
# passwd entry in here — give tools that want a HOME somewhere writable.
ENV HOME=/tmp \
    PYTHONUNBUFFERED=1

# Workdir is set to the repo root by run-ui.sh (-w "$SCRIPT_DIR"); app code is
# loaded from the mounted ./ui. Shell form so UI_HOST/UI_PORT interpolate.
# UI_TLS_ACTIVE is set by run-ui.sh only when a certificate actually exists
# (cert paths are repo-relative — the workdir is the repo root).
CMD exec python -m uvicorn --app-dir ui app:app \
      --host "${UI_HOST:-0.0.0.0}" --port "${UI_PORT:-8090}" --no-access-log \
      ${UI_TLS_ACTIVE:+--ssl-certfile "ui/data/certs/live/${UI_DOMAIN}/fullchain.pem" --ssl-keyfile "ui/data/certs/live/${UI_DOMAIN}/privkey.pem"}
