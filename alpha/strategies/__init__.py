import os
import cfg4py
import importlib
import inspect
import logging
from .base import BaseStrategy
import datetime

logger = logging.getLogger(__name__)


def get_all_strategies():
    cfg = cfg4py.get_instance()

    prefix = cfg.strategy.package_prefix
    if not prefix.endswith("."):
        prefix += "."

    source_dir = cfg.strategy.source_dir

    strategies = []

    exclude = cfg.strategy.exclude

    for file in os.listdir(source_dir):
        if file.endswith(".py") and file not in exclude:
            stem = file.replace(".py", "")
            module_name = prefix + stem

            try:
                module = importlib.import_module(module_name)
                classes = inspect.getmembers(module, inspect.isclass)
                for _, klz in classes:
                    if issubclass(klz, BaseStrategy) and klz.__module__ == module_name:
                        strategies.append((klz.name, klz.alias, klz.desc, klz.version, klz))
            except Exception as e:
                pass
                # logger.exception(e)

    return strategies


async def run_backtest(
    strategy_name: str,
    start: datetime.date,
    end: datetime.date,
    principal: float = 1_000_000,
    params: dict = None,
):
    strategies = get_all_strategies()

    for name, *_, klz in strategies:
        if name == strategy_name:
            strategy = klz()
            await strategy.start_backtest(start, end, principal, params)
    return None

async def on_remote_service_start():
    """the init function for strategy remote service
    """
    import cfg4py
    import omicron
    from pyemit import emit

    from alpha.config import get_config_dir

    logger.info("initializing process %s", os.getpid())
    cfg = cfg4py.init(get_config_dir())
    await omicron.init()
    await emit.start(emit.Engine.REDIS, start_server=True, dsn=cfg.redis.dsn)

async def on_remote_service_stop():
    """the cleanup function for strategy remote service"""
    import omicron
    from pyemit import emit

    logger.info("shutdown process %s", os.getpid())
    await omicron.close()
    await emit.stop()
