# vllm-pysandbox — ephemeral Python execution for the chat's run_python tool.
#
# One container per execution, always launched by ui/pyexec.py with:
#   --network none --read-only --tmpfs /tmp --memory 1g --cpus 2
#   --pids-limit 128 --user 65534:65534  and a 30s in-container timeout.
# The scientific stack is baked in because the sandbox has NO network —
# pip install at runtime is impossible by design.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
      numpy pandas matplotlib sympy scipy pillow networkx

COPY runner.py /runner.py

# MPLCONFIGDIR must be writable for the nobody user (tmpfs /tmp)
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
