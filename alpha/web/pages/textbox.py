import datetime
import logging
import re
from functools import partial
from typing import List

import arrow
from coretypes import Frame, FrameType
from h2o_wave import Expando, Q, ui
from omicron import tf
from omicron.models.stock import Stock
from plotly import io as pio

from alpha.web.widgets import inlinejs

from ..routing import StopPropagation, handle_on, on

logger = logging.getLogger(__name__)


@on("#textbox")
async def research_view(q: Q):
    if q.client.layout != "#textbox":
        await init_view(q)
    q.client.layout = "#textbox"


async def init_view(q: Q):
    q.page["meta"] = ui.meta_card(
        box="",
        title="Alpha",
        theme=q.user.theme,
        layouts=[
            ui.layout(
                breakpoint="xs",
                zones=[
                    ui.zone("header"),
                    ui.zone("body", zones=[ui.zone("box1"), ui.zone("box2")]),
                ],
            )
        ],
    )

    box1 = """
    <div id="box1">
        <input label='buy price' placeholder='please input stock price'/>
    </div>
    """
    q.page["box1"] = ui.markup_card(box="box1", title="box1", content=box1)

    box2 = """
    <div id="box2">
        <input label='sell price' placeholder='please input stock price'/>
    </div>
    """
    q.page["box2"] = ui.markup_card(box="box2", title="box2", content=box2)

    js = """
        var selector = "#box1 input"
        var event = "input"

        //uncomment this to let server handle the change
        //var callback = wave_emit("input", "box1", "value");

        var callback = function(){
            console.info("box2 is updated without server's interfere")
            box1_value = document.querySelector("#box1 input").value
            document.querySelector("#box2 input").value=box1_value
        }
        
        bind_event(selector, event, callback)
        console.info("bind event to:", selector, event)
    """
    q.page["meta"].script = inlinejs(js, targets=["#box1 input"])

    await q.page.save()


@on("input.box1")
async def change_box2(q: Q):
    print("change_box2 is called")
    price = q.events.input.box1

    js = f"""
    console.info("change box2's value as '{price}'")
    document.querySelector('#box2 input').value='{price}'
    """
    q.page["meta"].script = ui.inline_script(js)
    await q.page.save()
