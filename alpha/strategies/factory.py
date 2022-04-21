import importlib
import logging
import pkgutil

from alpha.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

reg = {}


def init_factory():
    for (module_name, name, ispkg) in pkgutil.iter_modules(["alpha/strategies"]):
        if name.startswith("_") or name == "base":
            continue
        importlib.import_module("." + name, package="alpha.strategies")

    for clz in BaseStrategy.__subclasses__():
        module = clz.__module__
        short_name = clz.__module__.split(".")[-1]
        clz_name = clz.__name__
        assert reg.get(short_name) is None

        reg[short_name] = f"{module}.{clz_name}"


def create_strategy(name: str, *args, **kwargs):
    """create a strategy object by its humanlized name

    Args:
        name (str): [description]
        args: positional args which should pass to constructor
        kwargs: key-word args which should pass to constructor
    """
    global reg

    if name not in reg:
        raise ValueError(f"strategy {name} is not registered")

    module_cls_name = reg[name]

    module_name, class_name = module_cls_name.rsplit(".", 1)
    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        raise NotImplementedError(
            f"module {module_name} doesn't contains stragety {class_name}"
        )

    cls = getattr(module, class_name)

    if not issubclass(cls, BaseStrategy):
        raise TypeError(f"{class_name} is not a subclass of {BaseStrategy}")

    obj = cls(*args, **kwargs)
    return obj


init_factory()
obj = create_strategy("grid-strategy")
print(obj)
