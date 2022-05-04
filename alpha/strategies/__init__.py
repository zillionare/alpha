import os
import cfg4py
import importlib
import inspect
import logging
from .base import BaseStrategy

logger = logging.getLogger(__name__)


def get_all_strategies():
    cfg = cfg4py.get_instance()

    prefix = cfg.alpha.strategy.package_prefix
    if not prefix.endswith("."):
        prefix += "."

    source_dir = cfg.alpha.strategy.source_dir

    strategies = []

    exclude = cfg.alpha.strategy.exclude

    for file in os.listdir(source_dir):
        if file.endswith(".py") and file not in exclude:
            stem = file.replace(".py", "")
            module_name = prefix + stem

            try:
                module = importlib.import_module(module_name)
                classes = inspect.getmembers(module, inspect.isclass)
                for _, klz in classes:
                    if issubclass(klz, BaseStrategy) and klz.__module__ == module_name:
                        strategies.append(
                            (klz.name, klz.desc, klz, klz.backtest_params)
                        )
            except Exception as e:
                pass
                # logger.exception(e)

    return strategies
