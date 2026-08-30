"""vLLM container management: run.sh parsing, docker inspection, launching.

The UI never re-implements model launch logic — it shells out to ./run.sh so
the empirically-verified per-model flags stay the single source of truth.
This module:

  * parses run.sh's select_model() case blocks (key → SNAPSHOT_REPO +
    default MODEL_ARGS) and COMMON_ARGS — a light structural parse; keep the
    run.sh layout (`key)` … `SNAPSHOT_REPO="…"` … `MODEL_ARGS=( … )` … `;;`)
    if you edit that file;
  * inspects the `vllm` container (running key via the vllm.model-key label,
    with a snapshot-path fallback for containers started by older run.sh);
  * launches/stops models: `env HOST_IP=127.0.0.1 ./run.sh <key> <overrides>`
    run as a background subprocess with output teed to ui/data/launch.log
    (weight downloads can block for many minutes);
  * scrapes vLLM /metrics and `docker exec vllm nvidia-smi` for the dashboard.

Runs inside the vllm-ui container with the docker socket and the repo mounted
at its identical host path, so the `docker run -v $SCRIPT_DIR/…` binds that
run.sh issues resolve correctly on the host daemon.
"""

import asyncio
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import state

REPO_DIR = Path(__file__).resolve().parent.parent
RUN_SH = REPO_DIR / "run.sh"
CONTAINER = "vllm"
MODEL_KEY_LABEL = "vllm.model-key"

# ─── run.sh parsing ─────────────────────────────────────────────────────────

def _parse_bash_array(text: str, name: str) -> list[str]:
    """Extract `name=( … )` items with shlex (handles quoted JSON values)."""
    m = re.search(rf"^{name}=\(\n(.*?)^\)", text, re.M | re.S)
    if not m:
        m = re.search(rf"{name}=\((.*?)\)", text, re.S)
        if not m:
            return []
    items: list[str] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.extend(shlex.split(line, comments=True))
    return items


def parse_run_sh() -> dict:
    """Return {key: {"repo": str, "model_args": [str], "speculator": str|None}}.

    Parses each `key)` branch of select_model(). Multi-key labels (`a|b)`) map
    every key to the same branch. Cross-checked against `run.sh --list` by the
    caller (list_models).
    """
    text = RUN_SH.read_text()
    m = re.search(r"^select_model\(\) \{\n(.*?)^\}", text, re.M | re.S)
    body = m.group(1) if m else text

    models: dict[str, dict] = {}
    # Split on case labels: a line like `    key)` or `    a|b)` (not `*)`).
    branches = re.split(r"^\s*([A-Za-z0-9._|-]+)\)\s*$", body, flags=re.M)
    # branches = [prefix, label1, body1, label2, body2, …]
    for label, branch in zip(branches[1::2], branches[2::2]):
        branch = branch.split(";;")[0]
        repo_m = re.search(r'SNAPSHOT_REPO="([^"]+)"', branch)
        if not repo_m:
            continue
        spec_m = re.search(r'SPECULATOR_REPO="([^"]+)"', branch)
        args = _parse_bash_array(branch, "MODEL_ARGS")
        for key in label.split("|"):
            models[key] = {
                "repo": repo_m.group(1),
                "model_args": args,
                "speculator": spec_m.group(1) if spec_m else None,
            }
    return models


def parse_common_args() -> list[str]:
    return _parse_bash_array(RUN_SH.read_text(), "COMMON_ARGS")


def run_sh_keys() -> list[str]:
    """Machine-readable key list straight from `./run.sh --list`."""
    out = subprocess.run(
        ["bash", str(RUN_SH), "--list"],
        capture_output=True, text=True, cwd=REPO_DIR, timeout=15,
    )
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def pi_models() -> dict:
    """{key: metadata} from pi.models.json (names, contextWindow, vision…)."""
    try:
        data = json.loads((REPO_DIR / "pi.models.json").read_text())
        entries = data["providers"]["vllm"]["models"]
        return {m["id"]: m for m in entries}
    except (OSError, KeyError, json.JSONDecodeError, ValueError):
        return {}


def fold_flags(args: list[str]) -> dict:
    """Fold a CLI arg list into {flag: value|True|[values]} — last occurrence
    wins, mirroring vLLM's argparse. Multi-value flags (--served-model-name)
    collect a list."""
    flags: dict = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith("--"):
            i += 1
            continue
        vals = []
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            vals.append(args[j])
            j += 1
        flags[tok] = True if not vals else (vals[0] if len(vals) == 1 else vals)
        i = j
    return flags


# ─── docker inspection ──────────────────────────────────────────────────────

async def _run(cmd: list[str], timeout: float = 20) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", "timeout"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def inspect_container() -> dict | None:
    rc, out, _ = await _run(["docker", "inspect", CONTAINER])
    if rc != 0:
        return None
    try:
        return json.loads(out)[0]
    except (json.JSONDecodeError, IndexError):
        return None


def container_model_key(info: dict, run_models: dict) -> str | None:
    """Which lineup key is this container serving?

    Primary: the vllm.model-key label run.sh stamps at launch.
    Fallback (pre-label containers): map Cmd[0]'s snapshot path back to a
    repo, then repo → key via the parsed run.sh table.
    """
    key = (info.get("Config", {}).get("Labels") or {}).get(MODEL_KEY_LABEL)
    if key:
        return key
    cmd = info.get("Config", {}).get("Cmd") or []
    if cmd:
        m = re.search(r"models--([^/]+)--([^/]+)/", cmd[0])
        if m:
            repo = f"{m.group(1)}/{m.group(2)}"
            for k, v in run_models.items():
                if v["repo"] == repo:
                    return k
    return None


def container_api_key(info: dict) -> str | None:
    """The VLLM_API_KEY the live container was launched with (docker inspect
    Env is authoritative — covers manual launches with a different .env key).
    None = the container's /v1 endpoints are unauthenticated."""
    for e in (info.get("Config", {}).get("Env") or []):
        if e.startswith("VLLM_API_KEY="):
            return e.split("=", 1)[1] or None
    return None


def container_upstream(info: dict) -> str | None:
    """Base URL the proxy should forward to, derived from the live container
    (works for containers we didn't start, whatever address they bound).
    0.0.0.0 → 127.0.0.1 (we run with --network host, loopback always reaches
    a published port). No published port → the container's bridge IP:8000,
    which the host network namespace routes to directly."""
    bindings = (info.get("HostConfig", {}).get("PortBindings") or {}).get("8000/tcp") or []
    if bindings:
        ip = bindings[0].get("HostIp") or "0.0.0.0"
        if ip in ("0.0.0.0", ""):
            ip = "127.0.0.1"
        return f"http://{ip}:{bindings[0].get('HostPort', '8080')}"
    ip = info.get("NetworkSettings", {}).get("IPAddress")
    if ip:
        return f"http://{ip}:8000"
    return None


# ─── host networks / serving bind ───────────────────────────────────────────

def host_interfaces() -> list[dict]:
    """IPv4 addresses of the host's interfaces (visible thanks to
    --network host). Loopback is skipped — it's already a preset."""
    out = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                         capture_output=True, text=True, timeout=10).stdout
    seen = []
    for line in out.splitlines():
        parts = line.split()
        # "2: eth0    inet 192.168.1.7/24 brd ..."
        if len(parts) >= 4 and parts[2] == "inet":
            ifname, ip = parts[1], parts[3].split("/")[0]
            if ifname != "lo" and ip not in [s["ip"] for s in seen]:
                seen.append({"ifname": ifname, "ip": ip})
    return seen


def env_file_defaults() -> dict:
    """HOST_IP / HOST_PORT / BIND_CIDR as written in the repo .env (display
    only — run.sh applies them itself when the launcher passes no override)."""
    vals = {}
    env_file = REPO_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("HOST_IP", "HOST_PORT", "BIND_CIDR"):
                vals[k.strip()] = v.strip().strip("\"'")
    return vals


# ─── launcher ───────────────────────────────────────────────────────────────

OVERRIDE_FIELDS = {
    # form field       flag                          type, min, max
    "max_model_len": ("--max-model-len", int, 1024, 262144),
    "gpu_memory_utilization": ("--gpu-memory-utilization", float, 0.5, 0.98),
    "max_num_seqs": ("--max-num-seqs", int, 1, 256),
    "max_num_batched_tokens": ("--max-num-batched-tokens", int, 256, 262144),
    "kv_cache_dtype": ("--kv-cache-dtype", str, None, None),
}
KV_DTYPES = {"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}


def overrides_to_args(ov: dict) -> list[str]:
    """Validate a stored override dict and render it to run.sh CLI args.
    Appended after the model key, these win over COMMON_ARGS and MODEL_ARGS
    (run.sh forwards them last; vLLM's argparse is last-wins).
    Raises ValueError on anything out of range."""
    args: list[str] = []
    for field, (flag, typ, lo, hi) in OVERRIDE_FIELDS.items():
        if field not in ov or ov[field] in (None, ""):
            continue
        try:
            val = typ(ov[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field}: not a valid {typ.__name__}")
        if typ is str:
            if val not in KV_DTYPES:
                raise ValueError(f"{field}: must be one of {sorted(KV_DTYPES)}")
        elif lo is not None and not (lo <= val <= hi):
            raise ValueError(f"{field}: must be within [{lo}, {hi}]")
        args += [flag, str(val)]
    extra = ov.get("extra_args") or ""
    if isinstance(extra, str):
        extra = shlex.split(extra)
    for tok in extra:
        if "\n" in tok or ";" in tok:
            raise ValueError(f"extra_args: rejected token {tok!r}")
    args += list(extra)
    return args


class Launcher:
    """Runs ./run.sh in the background, one launch at a time, and reports a
    phase machine for the frontend:
        preparing → downloading → loading → ready | failed
    State is never trusted from memory alone — callers combine this with a
    live docker inspect + /health probe (the watchdog or --restart may act on
    the container underneath us at any time)."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.current: dict | None = None   # {key, started_at, returncode, error}
        self.proc: asyncio.subprocess.Process | None = None

    def busy(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, key: str, override_args: list[str],
                    bind: dict | None = None, api_key: str | None = None) -> None:
        if self.busy():
            raise RuntimeError("a launch is already in progress")
        state.LAUNCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = open(state.LAUNCH_LOG, "wb")
        # Serving bind: an explicit UI choice is passed as real env vars, which
        # beat .env inside run.sh. With no UI choice the launcher passes
        # nothing and run.sh applies its own defaults (.env HOST_IP/BIND_CIDR/
        # HOST_PORT). Anything but 127.0.0.1 exposes raw, tokenless vLLM on
        # that interface — the UI warns, the proxy keeps working either way.
        env = dict(os.environ)
        bind = bind or {}
        if bind.get("host"):
            env["HOST_IP"] = bind["host"]
        if bind.get("port"):
            env["HOST_PORT"] = str(bind["port"])
        # The model port itself is never open tokenless: run.sh forwards
        # VLLM_API_KEY into the vLLM container, gating its /v1 endpoints with
        # the same bearer token as the proxy (env, not argv — keeps the secret
        # out of the launch log). Renewing the token in the UI applies to the
        # direct port on the NEXT model start.
        if api_key:
            env["VLLM_API_KEY"] = api_key
        bind_note = f" [bind {env.get('HOST_IP', '.env default')}:{env.get('HOST_PORT', '.env default')}]"
        log.write(f">>> [ui] launching {key} {' '.join(override_args)}{bind_note}\n".encode())
        log.flush()
        self.proc = await asyncio.create_subprocess_exec(
            "bash", str(RUN_SH), key, *override_args,
            cwd=REPO_DIR, stdout=log, stderr=log,
            env=env,
        )
        self.current = {"key": key, "started_at": time.time(),
                        "returncode": None, "error": None}
        asyncio.get_event_loop().create_task(self._reap(log))

    async def _reap(self, log) -> None:
        rc = await self.proc.wait()
        log.close()
        if self.current is not None:
            self.current["returncode"] = rc
            if rc != 0:
                self.current["error"] = f"run.sh exited with code {rc}"

    def log_tail(self, lines: int = 30) -> list[str]:
        try:
            return state.LAUNCH_LOG.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return []

    def phase(self, info: dict | None, health_ok: bool) -> dict | None:
        """Combine subprocess + container + health into one phase report."""
        cur = self.current
        if cur is None:
            return None
        elapsed = time.time() - cur["started_at"]
        report = {"key": cur["key"], "elapsed": int(elapsed),
                  "log_tail": self.log_tail(), "error": cur["error"]}
        if cur["returncode"] not in (None, 0):
            report["phase"] = "failed"
            return report
        if info is None:
            tail = "\n".join(report["log_tail"])
            report["phase"] = "downloading" if "download" in tail.lower() else "preparing"
            return report
        st = info.get("State", {})
        if health_ok:
            report["phase"] = "ready"
            self.current = None  # launch complete
            return report
        # Crash-loop: --restart unless-stopped resurrects a config that can't
        # boot forever. Surface it instead of showing "loading" for eternity.
        if st.get("RestartCount", 0) >= 2 or st.get("Status") == "restarting":
            report["phase"] = "failed"
            report["error"] = (report["error"] or
                               f"container is crash-looping (restarts: {st.get('RestartCount')}) — "
                               "check logs; likely OOM from an override. "
                               "Stop it and retry with defaults.")
            return report
        report["phase"] = "loading"
        return report


# ─── metrics / GPU ──────────────────────────────────────────────────────────

def parse_prometheus(text: str) -> dict:
    """Minimal Prometheus text-format parser → {metric: [(labels, value)]}."""
    out: dict[str, list] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)", line)
        if not m:
            continue
        name, labels_s, val_s = m.groups()
        try:
            val = float(val_s)
        except ValueError:
            continue
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', labels_s or ""))
        out.setdefault(name, []).append((labels, val))
    return out


def _first(metrics: dict, *names: str) -> float | None:
    for n in names:
        if n in metrics and metrics[n]:
            return metrics[n][0][1]
    return None


def _total(metrics: dict, *names: str) -> float | None:
    for n in names:
        if n in metrics and metrics[n]:
            return sum(v for _, v in metrics[n])
    return None


def _hist_stats(metrics: dict, base: str) -> dict | None:
    """mean from _sum/_count; p99 interpolated from cumulative buckets."""
    s, c = _total(metrics, base + "_sum"), _total(metrics, base + "_count")
    if not c:
        return None
    stats = {"mean": s / c, "count": c}
    buckets = metrics.get(base + "_bucket", [])
    if buckets:
        cum = sorted(((float("inf") if l.get("le") == "+Inf" else float(l["le"])), v)
                     for l, v in buckets)
        target = 0.99 * c
        for le, v in cum:
            if v >= target:
                stats["p99"] = le if le != float("inf") else None
                break
    return stats


def summarize_metrics(text: str) -> dict:
    """Curated dashboard view of vLLM's /metrics. Counter values are raw —
    the frontend computes deltas between polls for tokens/s."""
    m = parse_prometheus(text)
    out = {
        "ts": time.time(),
        "requests_running": _first(m, "vllm:num_requests_running"),
        "requests_waiting": _first(m, "vllm:num_requests_waiting"),
        # name drifted across vLLM versions — accept both
        "kv_cache_usage": _first(m, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        "prompt_tokens_total": _total(m, "vllm:prompt_tokens_total"),
        "generation_tokens_total": _total(m, "vllm:generation_tokens_total"),
        "requests_success_total": _total(m, "vllm:request_success_total"),
        "preemptions_total": _total(m, "vllm:num_preemptions_total"),
        "ttft": _hist_stats(m, "vllm:time_to_first_token_seconds"),
        "tpot": _hist_stats(m, "vllm:time_per_output_token_seconds"),
        "e2e_latency": _hist_stats(m, "vllm:e2e_request_latency_seconds"),
    }
    # EAGLE3 speculative decoding acceptance rate (gpt-oss)
    acc = _total(m, "vllm:spec_decode_num_accepted_tokens_total")
    draft = _total(m, "vllm:spec_decode_num_draft_tokens_total")
    if acc is not None and draft:
        out["spec_accept_rate"] = acc / draft
    return out


GPU_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"


async def gpu_stats() -> dict | None:
    """nvidia-smi via the vllm container (the toolkit injects the binary)."""
    rc, out, _ = await _run(
        ["docker", "exec", CONTAINER, "nvidia-smi",
         f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"])
    if rc != 0 or not out.strip():
        return None
    try:
        util, mem_used, mem_total, temp, power, plimit = \
            [float(x.strip()) for x in out.strip().splitlines()[0].split(",")]
    except (ValueError, IndexError):
        return None
    return {"utilization_pct": util, "memory_used_mib": mem_used,
            "memory_total_mib": mem_total, "temperature_c": temp,
            "power_w": power, "power_limit_w": plimit}


async def docker_logs(tail: int = 200) -> str:
    rc, out, err = await _run(["docker", "logs", "--tail", str(tail), CONTAINER])
    return out + err if rc == 0 else f"(no {CONTAINER} container)"


async def stop_container() -> bool:
    """Same semantics as ./stop-vllm.sh: remove the container entirely, so
    --restart unless-stopped cannot resurrect it."""
    rc, _, _ = await _run(["docker", "rm", "-f", CONTAINER], timeout=60)
    return rc == 0
