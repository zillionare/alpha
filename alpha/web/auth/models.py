import uuid
from collections import defaultdict
from typing import Any
import os

import cfg4py
import flask
from alpha.config import get_config_dir


class SessionStore:
    _users = {}
    _sessions = defaultdict(dict)

    def load_users(self):
        cfg = cfg4py.init(get_config_dir())
        with open(os.path.expanduser(cfg.auth.users_file), "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                name, password = line.split(":")
                self._users[name] = password

    def login_user(self, response, username, password):
        """call this in callback of login view only, since we used response"""
        sid = get_sid()
        if self._sessions.get(sid):
            return True

        if username is None or password is None:
            return False

        pwd = self._users.get(username, None)
        if pwd == password:
            sid = uuid.uuid4().hex
            self._sessions[sid] = {"name": username, "state": {}}
            response.set_cookie("zillionare-auth-session", sid)
            return True
        return False

    def get_user(self) -> str:
        """获取当前用户名"""
        sid = get_sid()

        return self._sessions.get(sid, {}).get("name", None)

    def remove_session(self):
        sid = get_sid()

        if sid in self._sessions:
            del self._sessions[sid]

    def save(self, key: str, value: Any):
        sid = get_sid()
        self._sessions[sid][key] = value

    def get(self, key: str, default: Any = None):
        sid = flask.request.cookies.get("zillionare-auth-session")

        return self._sessions.get(sid, {}).get(key, default)


sessions = SessionStore()
sessions.load_users()


def get_sid():
    return flask.request.cookies.get("zillionare-auth-session")


def get_user():
    return sessions.get_user()


def login_user(response, name, password):
    return sessions.login_user(response, name, password)


def remove_session():
    sessions.remove_session()


def save(key: str, value: Any):
    sessions.save(key, value)


def delete(key: str):
    save(key, None)


def get(key: str, default: Any = None):
    return sessions.get(key, default)


__all__ = [
    "sessions",
    "get_user",
    "login_user",
    "save",
    "get",
    "remove_session",
    "get_sid",
    "delete",
]
