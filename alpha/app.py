import logging

from h2o_wave import Expando, Q, app, main
from h2o_wave.server import _App

from alpha.web.pages import *
from alpha.web.pages.research import research_view
from alpha.web.routing import handle_on
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_on_client_connected = None

async def serve(q: Q):
    global _on_client_connected
    if not q.client.initialized and _on_client_connected:
        await _on_client_connected(q)
        q.client.initialized = True

    await handle_on(q)

def start(route: str, mode=None, on_app_startup: Optional[Callable] = None, on_app_shutdown: Optional[Callable] = None, on_client_connected: Optional[Callable] = None):
    """
    Start the application.

    Args:
        route: The route to listen to. e.g. `'/foo'` or `'/foo/bar/baz'`.
        mode: The server mode. One of `'unicast'` (default),`'multicast'` or `'broadcast'`.
        on_startup: A callback to invoke on app startup. Callbacks do not take any arguments, and may be be either standard functions, or async functions.
        on_shutdown: A callback to invoke on app shutdown. Callbacks do not take any arguments, and may be be either standard functions, or async functions.
    """
    global _on_client_connected

    _on_client_connected = on_client_connected
    main._app = _App(route, serve, mode, on_app_startup, on_app_startup)
