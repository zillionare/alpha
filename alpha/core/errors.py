class Error(BaseException):
    """Base class for all errors."""

    def __init__(self, msg: str) -> None:
        self.msg = msg


class NoTargetError(Error):
    pass


class NoFeaturesError(Error):
    pass


class TaskIsRunningError(Error):
    pass
