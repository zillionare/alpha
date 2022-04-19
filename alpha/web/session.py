from typing import Any

_session_store = {}


def save(sid: str, key: str, value: Any):
    state = _session_store.get(sid, {})
    state[key] = value

    _session_store[sid] = state


def get(sid: str, key: str, default: Any = None):
    if sid not in _session_store:
        return default

    state = _session_store.get(sid)

    return state.get(key, default)
