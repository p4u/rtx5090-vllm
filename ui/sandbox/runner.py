"""Sandbox entrypoint: execute /work/code.py, capture output, save figures.

Runs as the container command. Captures stdout+stderr, prints a JSON result
to fd 3? No — keeps it simple: writes /work/result.json. Any matplotlib
figures still open after the code ran are saved to /work/fig_N.png (the
ChatGPT-style convention: the user sees charts without the code needing an
explicit savefig)."""

import io
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout

OUT_CAP = 8000

out, err = io.StringIO(), io.StringIO()
ok = True
with open("/work/code.py") as f:
    code = f.read()
try:
    with redirect_stdout(out), redirect_stderr(err):
        exec(compile(code, "<chat>", "exec"), {"__name__": "__main__"})
except SystemExit:
    pass
except BaseException:
    ok = False
    err.write(traceback.format_exc(limit=8))

figures = 0
try:
    import matplotlib.pyplot as plt
    for i, num in enumerate(plt.get_fignums(), 1):
        plt.figure(num).savefig(f"/work/fig_{i}.png", dpi=110,
                                bbox_inches="tight")
        figures += 1
except Exception:
    pass

def cap(s: str) -> str:
    return s if len(s) <= OUT_CAP else s[:OUT_CAP] + f"\n…[truncated, {len(s)} chars total]"

with open("/work/result.json", "w") as f:
    json.dump({"ok": ok, "stdout": cap(out.getvalue()),
               "stderr": cap(err.getvalue()), "figures": figures}, f)
