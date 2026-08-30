"""Persistent UI state — a single JSON file under ui/data/.

Holds everything that must survive a UI-container restart:
  api_token       the single bearer token gating the OpenAI proxy (/v1/*)
  session_secret  signs the login session cookies (persisted so sessions
                  survive UI restarts)
  overrides       per-model launch-flag overrides set from the details drawer,
                  e.g. {"qwen3-coder": {"max_model_len": 131072, "extra_args": []}}
  last_model      last key launched from the UI (informational)

Writes are atomic (tmp + os.replace) so a crash mid-write never corrupts the
token. The file lives in the bind-mounted repo (ui/data/ is gitignored), so
state is owned by the host user and shared across UI container rebuilds.
"""

import json
import os
import secrets
import threading
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
STATE_FILE = DATA_DIR / "state.json"
LAUNCH_LOG = DATA_DIR / "launch.log"

_lock = threading.Lock()


def _default_state() -> dict:
    return {
        "api_token": secrets.token_urlsafe(32),
        "session_secret": secrets.token_urlsafe(32),
        "overrides": {},
        "last_model": None,
    }


def load() -> dict:
    with _lock:
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                state = {}
        else:
            state = {}
        # Fill any missing keys (first boot, or schema growth).
        changed = False
        for k, v in _default_state().items():
            if k not in state:
                state[k] = v
                changed = True
        if changed:
            _save_locked(state)
        return state


def save(state: dict) -> None:
    with _lock:
        _save_locked(state)


def _save_locked(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(tmp, STATE_FILE)


def update(**kwargs) -> dict:
    """Read-modify-write helper: load, apply kwargs, save, return new state."""
    state = load()
    state.update(kwargs)
    save(state)
    return state


def renew_token() -> str:
    token = secrets.token_urlsafe(32)
    update(api_token=token)
    return token
