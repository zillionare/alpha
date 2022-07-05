import asyncio
import datetime
import functools
import importlib
import inspect
import logging
import os
import uuid
from functools import partial

import cfg4py
from pyemit import emit

from alpha.core.commons import DataEvent
from alpha.core.const import E_BACKTEST, E_EXECUTOR_BACKTEST
from alpha.core.remote import RemoteService

from .base import BaseStrategy

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
                        strategies.append(
                            (klz.name, klz.alias, klz.desc, klz.version, klz)
                        )
            except Exception as e:
                pass
                # logger.exception(e)

    return strategies


def find_file_by_strategy(name: str):
    cfg = cfg4py.get_instance()

    prefix = cfg.strategy.package_prefix
    if not prefix.endswith("."):
        prefix += "."

    source_dir = cfg.strategy.source_dir

    exclude = cfg.strategy.exclude

    for file in os.listdir(source_dir):
        if file.endswith(".py") and file not in exclude:
            stem = file.replace(".py", "")
            module_name = prefix + stem

            try:
                module = importlib.import_module(module_name)
                classes = inspect.getmembers(module, inspect.isclass)
                for _, klz in classes:
                    if getattr(klz, "name", None) and name == klz.name:
                        return os.path.join(source_dir, file)
            except Exception as e:
                pass
                # logger.exception(e)

    return None


def create_strategy_by_name(name: str):
    strategies = get_all_strategies()

    for _, *_, klz in strategies:
        if name == klz.name:
            return klz()

    logger.warning("strategy %s not found", name)
    return None


async def run_backtest(
    strategy: str,
    start: datetime.date,
    end: datetime.date,
    principal: float = 1_000_000,
    params: dict = None,
    account: str = None,
    token: str = None,
):
    s = create_strategy_by_name(strategy)
    if s is None:
        logger.warning("strategy %s not found", strategy)

    return await s.start_backtest(start, end, principal, params, account, token)


async def on_backtest_progress(event, msg):
    event.set(msg)


async def run_backtest_remote(
    strategy: str,
    start: datetime.date,
    end: datetime.date,
    principal: float = 1_000_000,
    params: dict = None,
    account: str = None,
    token: str = None,
):
    from alpha.core.executor import create_process_pool, process_pool

    if len(process_pool) == 0:
        await create_process_pool()

    event = DataEvent()

    emit.register(E_BACKTEST, partial(on_backtest_progress, event))
    await emit.emit(
        E_EXECUTOR_BACKTEST,
        {
            "strategy": strategy,
            "start": start,
            "end": end,
            "principal": principal,
            "params": params,
            "account": account,
            "token": token,
        },
    )

    return event
