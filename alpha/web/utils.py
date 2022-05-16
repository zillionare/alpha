import logging
import os
import re
from typing import List

from omicron.models.stock import Stock

logger = logging.getLogger(__name__)


def make_stock_input_hint(code: str) -> List[str]:
    if code is None:
        return []

    matched = Stock.fuzzy_match(code)
    #  ('000001.XSHE', '平安银行', 'PAYH'... 'stock')

    options = []

    if re.match(r"\d+", code):  # 用户输入了代码
        for v in matched.values():
            options.append(f"{v[0]} {v[1]}")
            if len(options) > 10:
                break
    elif re.match(r"[a-z]+", code.lower()):
        for v in matched.values():
            options.append(f"{v[2]} {v[1]}")
            if len(options) > 10:
                break
    else:
        for v in matched.values():
            options.append(f"{v[0]} {v[1]}")
            if len(options) > 10:
                break

    return options
