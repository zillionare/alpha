from alpha.core import strategyfactory
from omicron.core.types import FrameType

# aync multi processing

# strategy multi-processing model

"""
using cli to start workers. Each work can schedule tasks and execute tasks. Tasks will be saved into cache. Worker is listening on events and then check if there's new task

Task can be backtesting or prediction
"""


def main(strategy: str, frame_type: str):
    frame_type = FrameType(frame_type)
