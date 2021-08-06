"""Console script for alpha."""

import asyncio
import functools
import pickle
import sys

import cfg4py
import fire
from mergedeep import Strategy
import omicron
import psutil
from omicron import cache
from omicron.models.securities import Securities
from xgboost import XGBRegressor
import importlib.util

from alpha.strategies.z03 import Z03


def async_run_command(func):
    async def _init_and_run(*args, **kwargs):
        cfg4py.init("/apps/alpha/alpha/config")
        await omicron.init()
        await func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        asyncio.run(_init_and_run(*args, **kwargs))

    return wrapper


def help():
    print("alpha")
    print("=" * len("alpha"))
    print("Skeleton project created by Python Project Wizard (ppw)")


async def backtest(strategy: str, code: str = None, frame_type: str = "1d"):
    cpus = psutil.get_cpus()

    secs = Securities.choose(["stock"])
    await cache.sys.lpush(f"backtest.scope.{strategy}.{frame_type.value}", secs)
    for i in range(cpus):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "alpha.core.mp", "main", strategy, frame_type
        )


def create_strategy(strategy: str):
    spec = importlib.util.find_spec("alpha.strategies.{}".format(strategy.lower()))
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)

    ctor = getattr(strategy_module, strategy)
    return ctor()


@async_run_command
async def prepare_train_data(strategy: str, save_to: str, *args, **kwargs):
    s = create_strategy(strategy)
    data = await s.extract_features(*args, **kwargs)

    with open(save_to, "wb") as f:
        pickle.dump(data, f)


@async_run_command
async def train(strategy: str, save_to: str):
    if strategy == "z03":
        s = Z03()
        params = {
            "colsample_bytree": 0.722213395520227,
            "gamma": 0.1792328642721363,
            "learning_rate": 0.1458690595251297,
            "max_depth": 4,
            "n_estimators": 108,
            "subsample": 0.8493192507310232,
        }

        model = s.train(**params)

        with open(save_to, "wb") as f:
            pickle.dump(model, f)


@async_run_command
async def gridsearch(strategy: str, dataset_file: str):
    s = create_strategy(strategy)
    s.load_data(dataset_file)
    s.grid_search()


def main():
    fire.Fire(
        {
            "help": help,
            "train_data": prepare_train_data,
            "train": train,
            "gridsearch": gridsearch,
        }
    )


if __name__ == "__main__":
    main()  # pragma: no cover
