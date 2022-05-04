from h2o_wave import main, app, Q, ui, on, handle_on, data
from typing import Optional, List
from alpha.web.layout import meta, add_card, clear_cards
from alpha.web.pages.research import research_view
import omicron
import cfg4py
from alpha.config import get_config_dir


async def on_client_connected(q: Q) -> None:
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
    await q.page.save()
