from . import data


def round(x: float, decimal: int):
    return int(x * (10 ** decimal) + 0.5) / (10 ** decimal)


__all__ = ["data", "round"]
