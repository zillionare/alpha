import logging
import os
import re
from typing import List

from h2o_wave import Component, Q, ui
from omicron.models.stock import Stock

logger = logging.getLogger(__name__)

js_file = os.path.join(os.path.dirname(__file__), "assets/js/util.js")
with open(js_file, "r") as f:
    js = f.read()
    
def make_stock_input_hint(code: str) -> List[str]:
    if code is None:
        return []

    matched = Stock.fuzzy_match(code)
    #  ('000001.XSHE', '平安银行', 'PAYH'... 'stock')

    options = []

    if re.match(r"\d+", code):  # 用户输入了代码
        for v in matched.values():
            code = v[0].split(".")[0]
            options.append(f"{code} - {v[1]}")
            if len(options) > 10:
                break
    elif re.match(r"[a-z]+", code.lower()):
        for v in matched.values():
            options.append(f"{v[2]} - {v[1]}")
            if len(options) > 10:
                break
    else:
        for v in matched.values():
            options.append(f"{v[1]}")
            if len(options) > 10:
                break

    return options


def inlinejs(ps: str, requires: List[str]=None, targets: List[str]=None) -> Component:
    """生成inline js

    Args:
        ps : page ad-hoc script

    Returns:
        ui.inline_script
    """
    global js
    return ui.inline_script(
            content = js + "\n" + ps,
            requires = requires,
            targets = targets,
        )
