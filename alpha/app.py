import logging
from typing import List, Optional

import cfg4py
import omicron
from h2o_wave import Q, app, data, handle_on, main, on, ui

from alpha.config import get_config_dir
from alpha.web.layout import add_card, clear_cards, meta
from alpha.web.pages.research import research_view
from alpha.web.utils import inject_util_js

logger = logging.getLogger(__name__)


async def on_client_connected(q: Q) -> None:
    inject_util_js(q)
    q.user.theme = q.user.theme or "ember"
    # If no active hash present, render research.
    if q.args["#"] is None:
        await research_view(q)


async def on_startup():
    cfg4py.init(get_config_dir())
    await omicron.init()


async def on_shutdown():
    await omicron.stop()


@app("/", on_startup=on_startup, on_shutdown=on_shutdown)
async def serve(q: Q):
    if not q.client.initialized:
        await on_client_connected(q)
        q.client.initialized = True

    # Handle routing.
    await handle_on(q)
