"""Console script for alpha."""

from warnings import simplefilter
from alpha.strategies.databunch import load_data
import asyncio
import functools
import pickle
import sys

import cfg4py
import fire
import omicron
import psutil
from omicron import cache
from omicron.models.securities import Securities
import importlib.util
import arrow



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
async def make_dataset(strategy: str, save_to: str, *args, **kwargs):
    s = create_strategy(strategy)
    bunch = await s.make_dataset(*args, **kwargs)

    with open(save_to, "wb") as f:
        pickle.dump(bunch, f)


@async_run_command
async def train(strategy: str, data_file:str, version:str=None):
    version = version or str(arrow.now().date())
    s = create_strategy(strategy)
    s.version = version

    ds = load_data(data_file)

    s.fit(ds)

def main():
    simplefilter("ignore")

    fire.Fire(
        {
            "help": help,
            "ds": make_dataset,
            "train": train,
        }
    )


if __name__ == "__main__":
    main()  # pragma: no cover
