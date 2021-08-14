"""Console script for alpha."""

from alpha.utils.data import even_distributed_dataset
import asyncio
import functools
import importlib.util
import os
import pickle
import sys
from warnings import simplefilter
import numpy as np

import arrow
import cfg4py
import fire
import omicron
import psutil
from genericpath import exists
from omicron import cache
from omicron.models.securities import Securities

from alpha.strategies.databunch import load_data


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
async def make_dataset(
    strategy: str, version: str = None, notes: str = "", *args, **kwargs
):
    s = create_strategy(strategy)

    version = version or str(arrow.now().date())
    bunch = await s.make_dataset(version=version, *args, **kwargs)
    bunch.desc = f"{strategy.lower()}.{version}:{notes}"

    home = os.path.expanduser(s.data_home)

    save_to = os.path.join(home, version, f"{s.name.lower()}.{version}.ds")
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, "wb") as f:
        pickle.dump(bunch, f)


@async_run_command
async def train(strategy: str, version: str = None, ds: str = None):
    version = version or str(arrow.now().date())
    s = create_strategy(strategy)
    s.version = version

    if ds is None:
        home = os.path.expanduser(s.data_home)
        data_file = os.path.join(home, version, f"{s.name.lower()}.{version}.ds")
    else:
        data_file = ds

    ds = load_data(data_file)

    s.fit(ds)


@async_run_command
async def make_even_distributed_dataset(total: int, save_to: str, bars_len: int = 300):
    buckets_size = 21

    def target_to_bucket(bars):
        c0, c1 = bars[-2:]["close"]

        if np.all(np.isfinite((c0, c1))):
            return c1 / c0 - 1, int((c1 / c0 - 1) * 100) + buckets_size // 2

        return None, None

    meta = {"target_win": 1}

    await even_distributed_dataset(
        total,
        buckets_size,
        bars_len,
        target_to_bucket,
        save_to,
        meta=meta,
        start="2015-10-09",
    )


def main():
    simplefilter("ignore")

    fire.Fire(
        {
            "help": help,
            "ds": make_dataset,
            "train": train,
            "make_even_ds": make_even_distributed_dataset,
        }
    )


if __name__ == "__main__":
    main()  # pragma: no cover
