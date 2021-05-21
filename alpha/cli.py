"""Console script for alpha."""

import asyncio
import sys

import fire
import psutil
from omicron import cache
from omicron.models.securities import Securities


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


def main():
    fire.Fire({"help": help})


if __name__ == "__main__":
    main()  # pragma: no cover
