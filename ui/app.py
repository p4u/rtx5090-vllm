"""vllm-ui — password-gated web UI + token-gated OpenAI reverse proxy.

Three route classes, three gates:
  /v1/*            Bearer-token gate (the single API token from state.json).
                   Streaming reverse proxy to whatever vLLM container runs.
  /api/*, /        session-cookie gate (login with UI_PASSWORD).
  /auth/login,
  /ui-health,
  static assets    open (the SPA shell is harmless without a session).

No TLS here — front it with a VPN or a TLS reverse proxy; password and token
travel in plaintext otherwise.
"""

import asyncio
import hmac
import json
import os
import sys
import time

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, TimestampSigner
from starlette.background import BackgroundTask

import state
import vllm_mgr

UI_PASSWORD = os.environ.get("UI_PASSWORD", "")
if not UI_PASSWORD:
    sys.exit("vllm-ui: UI_PASSWORD is required (set it in .env or the environment)")

SESSION_COOKIE = "vllm_ui_session"
SESSION_MAX_AGE = 7 * 24 * 3600

app = FastAPI(title="vllm-ui", docs_url=None, redoc_url=None, openapi_url=None)
launcher = vllm_mgr.Launcher()
_signer = TimestampSigner(state.load()["session_secret"])

# Upstream client for the OpenAI proxy and health/metrics probes.
# read=None is mandatory: long generations / SSE streams exceed any default.
client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5, read=None, write=None, pool=None))


# ─── auth helpers ───────────────────────────────────────────────────────────

def _session_ok(request: Request) -> bool:
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return False
    try:
        _signer.unsign(cookie, max_age=SESSION_MAX_AGE)
        return True
    except BadSignature:
        return False


def require_session(request: Request) -> None:
    if not _session_ok(request):
        raise HTTPException(401, "not logged in")


def require_token(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if not token or not hmac.compare_digest(token, state.load()["api_token"]):
        raise HTTPException(401, "invalid or missing API token")


@app.post("/auth/login")
async def login(request: Request):
    body = await request.json()
    password = str(body.get("password", ""))
    if not hmac.compare_digest(password.encode(), UI_PASSWORD.encode()):
        await asyncio.sleep(0.5)  # blunt brute-force throttle
        raise HTTPException(401, "wrong password")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(SESSION_COOKIE, _signer.sign(b"ok").decode(),
                    max_age=SESSION_MAX_AGE, httponly=True, samesite="lax")
    return resp


@app.post("/auth/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ─── upstream helpers ───────────────────────────────────────────────────────

async def _upstream() -> tuple[str | None, dict | None]:
    """(base_url, inspect_info) of the live vllm container, or (None, None)."""
    info = await vllm_mgr.inspect_container()
    if info is None:
        return None, None
    return vllm_mgr.container_upstream(info), info


async def _health_ok(base: str | None) -> bool:
    if not base:
        return False
    try:
        r = await client.get(base + "/health", timeout=3)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# ─── management API (session-gated) ─────────────────────────────────────────

@app.get("/api/state")
async def api_state(request: Request):
    require_session(request)
    run_models = vllm_mgr.parse_run_sh()
    base, info = await _upstream()
    healthy = await _health_ok(base)
    launch = launcher.phase(info, healthy)
    result = {
        "running_key": vllm_mgr.container_model_key(info, run_models) if info else None,
        "container": None,
        "healthy": healthy,
        "upstream": base,
        "launch": launch,
        "launching": launcher.busy() or (launch or {}).get("phase") in ("preparing", "downloading", "loading"),
    }
    if info:
        st = info.get("State", {})
        result["container"] = {
            "status": st.get("Status"),
            "health": (st.get("Health") or {}).get("Status"),
            "restart_count": st.get("RestartCount", 0),
            "started_at": st.get("StartedAt"),
            # Effective flags actually in force (last-wins fold of the real
            # container command — includes any overrides).
            "effective_flags": {
                k: v for k, v in vllm_mgr.fold_flags(
                    (info.get("Config", {}).get("Cmd") or [])[1:]).items()
                if k != "--served-model-name"  # 30+ aliases, noise
            },
        }
    return result


@app.get("/api/models")
async def api_models(request: Request):
    require_session(request)
    run_models = vllm_mgr.parse_run_sh()
    keys = vllm_mgr.run_sh_keys() or list(run_models)
    meta = vllm_mgr.pi_models()
    overrides = state.load()["overrides"]
    common = vllm_mgr.fold_flags(vllm_mgr.parse_common_args())
    models = []
    for key in keys:
        rm = run_models.get(key, {})
        defaults = {**common, **vllm_mgr.fold_flags(rm.get("model_args", []))}
        pm = meta.get(key, {})
        models.append({
            "key": key,
            "name": pm.get("name", key),
            "repo": rm.get("repo"),
            "context_window": pm.get("contextWindow"),
            "reasoning": pm.get("reasoning", False),
            "vision": "image" in pm.get("input", []),
            "defaults": {
                "max_model_len": defaults.get("--max-model-len"),
                "gpu_memory_utilization": defaults.get("--gpu-memory-utilization"),
                "max_num_seqs": defaults.get("--max-num-seqs"),
                "max_num_batched_tokens": defaults.get("--max-num-batched-tokens"),
                "kv_cache_dtype": defaults.get("--kv-cache-dtype"),
            },
            "default_args": rm.get("model_args", []),
            "overrides": overrides.get(key, {}),
        })
    return {"models": models}


@app.post("/api/models/{key}/start")
async def api_start(key: str, request: Request):
    require_session(request)
    run_models = vllm_mgr.parse_run_sh()
    if key not in run_models:
        raise HTTPException(404, f"unknown model key: {key}")
    if launcher.busy():
        raise HTTPException(409, "a launch is already in progress")
    body = await request.json() if int(request.headers.get("content-length") or 0) else {}
    overrides = body.get("overrides", {})
    try:
        override_args = vllm_mgr.overrides_to_args(overrides)
    except ValueError as e:
        raise HTTPException(422, str(e))
    st = state.load()
    st["overrides"][key] = overrides
    st["last_model"] = key
    state.save(st)
    await launcher.start(key, override_args)
    return {"ok": True, "key": key, "args": override_args}


@app.post("/api/stop")
async def api_stop(request: Request):
    require_session(request)
    launcher.current = None
    ok = await vllm_mgr.stop_container()
    return {"ok": ok}


@app.get("/api/logs")
async def api_logs(request: Request, tail: int = 200):
    require_session(request)
    return {"logs": await vllm_mgr.docker_logs(min(tail, 2000))}


@app.get("/api/logs/stream")
async def api_logs_stream(request: Request):
    """SSE wrapper around `docker logs -f` for the live log panel."""
    require_session(request)
    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", "50", vllm_mgr.CONTAINER,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    async def gen():
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                yield f"data: {json.dumps(line.decode(errors='replace').rstrip())}\n\n"
        finally:
            if proc.returncode is None:
                proc.kill()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/metrics")
async def api_metrics(request: Request):
    require_session(request)
    base, _ = await _upstream()
    if not base:
        return {"available": False}
    try:
        r = await client.get(base + "/metrics", timeout=5)
        r.raise_for_status()
    except httpx.HTTPError:
        return {"available": False}
    return {"available": True, **vllm_mgr.summarize_metrics(r.text)}


@app.get("/api/gpu")
async def api_gpu(request: Request):
    require_session(request)
    stats = await vllm_mgr.gpu_stats()
    return {"available": stats is not None, **(stats or {})}


@app.get("/api/token")
async def api_token(request: Request):
    require_session(request)
    return {"token": state.load()["api_token"]}


@app.post("/api/token/renew")
async def api_token_renew(request: Request):
    require_session(request)
    return {"token": state.renew_token()}


# ─── chat (session-gated passthrough for the built-in chat tab) ─────────────

@app.post("/api/chat")
async def api_chat(request: Request):
    """Streaming chat completion for the UI's chat tab. Same upstream as the
    /v1 proxy but gated by the login session instead of the API token, so the
    browser never handles the bearer token."""
    require_session(request)
    base, _ = await _upstream()
    if not base:
        return JSONResponse(
            {"error": {"message": "no model is running — start one from the Models tab",
                       "type": "service_unavailable"}}, status_code=503)
    body = await request.body()
    upstream_req = client.build_request(
        "POST", f"{base}/v1/chat/completions",
        headers={"Content-Type": "application/json"}, content=body)
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.HTTPError:
        return JSONResponse(
            {"error": {"message": "model is booting or unreachable — try again shortly",
                       "type": "service_unavailable"}}, status_code=503)
    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() not in HOP_BY_HOP}
    return StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code,
        headers=resp_headers, background=BackgroundTask(upstream.aclose))


# ─── OpenAI reverse proxy (token-gated) ─────────────────────────────────────

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "upgrade",
              "proxy-authenticate", "proxy-authorization", "te", "trailer", "host"}


@app.api_route("/v1/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    require_token(request)
    base, info = await _upstream()
    if not base:
        return JSONResponse(
            {"error": {"message": "no model is running — start one from the UI",
                       "type": "service_unavailable"}}, status_code=503)
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in HOP_BY_HOP and k.lower() != "authorization"}
    upstream_req = client.build_request(
        request.method, f"{base}/v1/{path}",
        headers=headers, params=request.query_params,
        content=request.stream())
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.HTTPError:
        return JSONResponse(
            {"error": {"message": "model is booting or unreachable — try again shortly",
                       "type": "service_unavailable"}}, status_code=503)
    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() not in HOP_BY_HOP}
    # aiter_raw passes bytes through untouched (no decompress/re-buffer), so
    # SSE chunks flush incrementally and content-encoding stays valid.
    return StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code,
        headers=resp_headers, background=BackgroundTask(upstream.aclose))


# ─── static frontend ────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/ui-health")
async def ui_health():
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
