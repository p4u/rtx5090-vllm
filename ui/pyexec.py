"""Python execution tool for the chat (run_python).

One throwaway docker container per execution (same pattern as the browsing
tools): image `vllm-pysandbox` (built by run-ui.sh — python + numpy/pandas/
matplotlib/scipy/sympy/pillow/networkx), launched with NO network, read-only
rootfs, tmpfs /tmp, 1 GB / 2 cpus / 128 pids caps, as nobody, with a 30s
in-container timeout. The only writable bind is a per-run temp dir where the
code goes in and stdout/figures come out. pip install at runtime is
impossible by design (no network) — the library set is baked into the image.

Figures left open by the code are auto-saved (runner.py) and returned as
data-URL PNGs for the chat to display; the model only gets text (stdout,
stderr, figure count) — pixels go to the human.
"""

import base64
import json
import os
import tempfile
import time

import vllm_mgr

IMAGE = "vllm-pysandbox"
EXEC_TIMEOUT_S = 30
OUTER_TIMEOUT_S = 45
MAX_CODE_CHARS = 40000
MAX_FIGURES = 4

TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python code in a sandbox and return stdout/stderr. "
            "Available: numpy, pandas, matplotlib, scipy, sympy, pillow, "
            "networkx. NO network and NO pip install. To show the user a "
            "chart or diagram, create a matplotlib figure — open figures are "
            "captured and displayed automatically (no plt.show/savefig "
            "needed). State does not persist between calls; each call is a "
            "fresh interpreter."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
    },
}

_avail_cache: tuple[float, bool] | None = None


async def available() -> bool:
    global _avail_cache
    now = time.time()
    if _avail_cache and now - _avail_cache[0] < 60:
        return _avail_cache[1]
    rc, _, _ = await vllm_mgr._run(["docker", "image", "inspect", IMAGE], timeout=10)
    _avail_cache = (now, rc == 0)
    return rc == 0


async def run_python(args: dict) -> dict:
    code = str(args.get("code", ""))
    if not code.strip():
        return {"error": "empty code", "debug": []}
    if len(code) > MAX_CODE_CHARS:
        return {"error": f"code longer than {MAX_CODE_CHARS} chars", "debug": []}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="pyexec-",
                                     dir=str(vllm_mgr.REPO_DIR / "ui" / "data")) as td:
        os.chmod(td, 0o777)               # the sandbox runs as nobody
        with open(os.path.join(td, "code.py"), "w") as f:
            f.write(code)
        rc, out, err = await vllm_mgr._run(
            ["docker", "run", "--rm",
             "--network", "none", "--read-only", "--tmpfs", "/tmp",
             "--memory", "1g", "--cpus", "2", "--pids-limit", "128",
             "--user", "65534:65534",
             "-v", f"{td}:/work",
             IMAGE, "timeout", str(EXEC_TIMEOUT_S), "python", "/runner.py"],
            timeout=OUTER_TIMEOUT_S)
        dur = round(time.time() - t0, 1)
        debug = [f"run_python exit={rc} dur={dur}s code_chars={len(code)}"]
        print(f"[pyexec] {debug[0]}", flush=True)
        result_path = os.path.join(td, "result.json")
        if not os.path.exists(result_path):
            msg = ("execution timed out after "
                   f"{EXEC_TIMEOUT_S}s" if rc == 124 else
                   f"sandbox failed (exit {rc}): {err.strip()[-200:]}")
            return {"error": msg, "duration": dur, "debug": debug}
        with open(result_path) as f:
            result = json.load(f)
        images = []
        for i in range(1, MAX_FIGURES + 1):
            p = os.path.join(td, f"fig_{i}.png")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    images.append("data:image/png;base64," +
                                  base64.b64encode(f.read()).decode())
        payload = {"ok": result["ok"], "stdout": result["stdout"],
                   "stderr": result["stderr"], "duration": dur, "debug": debug}
        if result["figures"]:
            # pixels for the human; the model just learns they exist
            payload["images"] = images
            payload["figures_note"] = (f"{len(images)} figure(s) rendered and "
                                       "displayed to the user")
        if not result["ok"]:
            payload["error"] = "code raised an exception (see stderr)"
        return payload
