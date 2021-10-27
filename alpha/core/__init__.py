from typing import NewType
import datetime
from arrow import Arrow

Frame = NewType("Frame", (str, datetime.datetime, datetime.date, Arrow))
