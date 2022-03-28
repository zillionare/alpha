import datetime
from typing import NewType

from arrow import Arrow

Frame = NewType("Frame", (str, datetime.datetime, datetime.date, Arrow))
