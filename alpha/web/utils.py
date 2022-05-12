import logging
import os
import re
from typing import List

from h2o_wave import Q, ui
from omicron.models.stock import Stock

logger = logging.getLogger(__name__)


def make_stock_input_hint(code: str) -> List[str]:
    if code is None:
        return []

    matched = Stock.fuzzy_match(code)[:10]
    #  ('000001.XSHE', '平安银行', 'PAYH'... 'stock')

    options = []

    if re.match(r"\d+", code):  # 用户输入了代码
        for v in matched.values():
            code = v[0].split(".")[0]
            options.append(f"{code} - {v[1]}")
    elif re.match(r"[a-z]+", code.lower()):
        for v in matched.values():
            options.append(f"{v[2]} - {v[1]}")
    else:
        for v in matched.values():
            options.append(f"{v[1]}")

    return options


def inject_util_js(q: Q) -> None:
    js_file = os.path.join(os.path.dirname(__file__), "assets/js/util.js")
    with open(js_file, "r") as f:
        js = f.read()
        q.page["meta"].script = ui.meta_card(
            box='',
            script = ui.inline_script(
                content = "console.log('hello world')",
            )
        )
