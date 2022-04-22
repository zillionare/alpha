import logging
import re
from typing import List
from asgiref.sync import async_to_sync
from coretypes import Frame, FrameType
from omicron.models.stock import Stock
from dash import html
from dash.development.base_component import Component

import dash

logger = logging.getLogger(__name__)


def get_triggerred_controls() -> List[str]:
    """获取触发的控件id，只能在dash回调函数中使用"""
    controls = []
    for control in dash.callback_context.triggered:
        try:
            controls.append(control["prop_id"].split(".")[0])
        except Exception as e:
            logger.exception(e)

    return controls


@async_to_sync
async def get_bars(
    stock: str, end: Frame, n: int, frame_type: FrameType, fq=True, unclosed=True
):
    return await Stock.get_bars(stock, n, frame_type, end, fq, unclosed)


def make_stock_input_hint(code: str) -> List[Component]:
    if code is None:
        return []

    matched = Stock.fuzzy_match(code)[:10]
    #  ('000001.XSHE', '平安银行', 'PAYH'... 'stock')

    options = []

    if re.match(r"\d+", code):  # 用户输入了代码
        for v in matched.values():
            code = v[0].split(".")[0]
            options.append(html.Option(v[0], label=f"{code} {v[1]}"))
    elif re.match(r"[a-z]+", code.lower()):
        for v in matched.values():
            options.append(html.Option(v[0], label=f"{v[2]} {v[1]}"))
    else:
        for v in matched.values():
            options.append(html.Option(v[0], label=f"{v[1]}"))

    if len(options) < 2: # fix display bug
        options.append(html.Option("", label=""))
    return options
