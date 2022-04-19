from re import L
from typing import List
import dash
import logging

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
