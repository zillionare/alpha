"""Console script for alpha."""

import asyncio
import functools
import importlib.util
import os
import pickle
import sys
from warnings import simplefilter

import arrow
import cfg4py
import fire
import numpy as np
import omicron
import psutil
from omicron import cache
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from aioproc import aiofunc

from alpha.utils.data import even_distributed_dataset, load_data


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
        await asyncio.create_subprocess_exec(
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
async def make_even_distributed_dataset(
    total: int,
    save_to: str,
    bars_len: int = 300,
    frame_type: str = "1d",
    start="2020-1-4 15:00",
    has_register_ipo=False,
):
    bins = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
    buckets = [0] * (len(bins) + 1)
    ylen = 10

    def target_to_bucket(bars):
        close = bars["close"]
        c0 = close[-ylen - 1]
        yclose = close[-ylen:]

        if np.isfinite(c0) and np.count_nonzero(np.isfinite(yclose)) > 0.9 * len(
            yclose
        ):
            max_adv = max(yclose) / c0 - 1
            max_dec = abs(min(yclose) / c0 - 1)

            agg = max if max_adv > max_dec else min
            y = agg(yclose) / c0 - 1

            for i, bin in enumerate(bins):
                if y < bin:
                    return y, i
            else:
                return y, i + 1
        else:
            return None, None

    reg = "reg" if has_register_ipo else "noreg"
    file = os.path.join(
        save_to, f"ds_even_{frame_type}_{ylen}_{bars_len}_{total}_{reg}.pkl"
    )
    meta = {
        "target_win": ylen,
        "frame_type": frame_type,
        "bins": bins,
        "bars_len": bars_len,
        "total": total,
        "start": start,
    }

    await even_distributed_dataset(
        total,
        buckets,
        bars_len,
        target_to_bucket,
        file,
        meta=meta,
        start=start,
        frame_type=FrameType(frame_type),
    )


@async_run_command
async def mpscan(strategy: str, *args, **kwargs):
    secs = Securities().choose(["stock"])
    await cache.sys.delete(f"scan.scope.{strategy}")
    await cache.sys.delete(f"scan.result.{strategy}")
    await cache.sys.lpush(f"scan.scope.{strategy}", *secs)

    procs = []
    for i in range(20):
        procs.append(aiofunc(
            "alpha.strategies", "scan", args=(strategy, *args), kwargs= kwargs, delay=2
        ))

    results = await asyncio.gather(*procs)
    print(results)

def main():
    simplefilter("ignore")

    fire.Fire(
        {
            "help": help,
            "ds": make_dataset,
            "train": train,
            "make_even_ds": make_even_distributed_dataset,
            "mpscan": mpscan
        }
    )


if __name__ == "__main__":
    main()  # pragma: no cover
