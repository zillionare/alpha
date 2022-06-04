"""Data and utilities for testing."""
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
from contextlib import closing

import aiohttp
import cfg4py
import numpy as np
import pandas as pd
from omicron.models.stock import Stock
from alpha.config import get_config_dir

cfg = cfg4py.get_instance()
logger = logging.getLogger()


def init_test_env():
    os.environ[cfg4py.envar] = "DEV"

    return cfg4py.init(get_config_dir(), False)


def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port


async def is_local_omega_alive(port: int = 3181):
    try:
        url = f"http://localhost:{port}/sys/version"
        async with aiohttp.ClientSession() as client:
            async with client.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
        return True
    except Exception:
        return False


async def start_omega(timeout=60):
    port = find_free_port()

    cfg.omega.urls.quotes_server = f"http://localhost:{port}"
    account = os.environ["JQ_ACCOUNT"]
    password = os.environ["JQ_PASSWORD"]

    # hack: by default postgres is disabled, but we need it enabled for ut
    cfg_ = json.dumps({"postgres": {"dsn": cfg.postgres.dsn, "enabled": "true"}})

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omega.app",
            "start",
            "--impl=jqadaptor",
            f"--cfg={cfg_}",
            f"--account={account}",
            f"--password={password}",
            f"--port={port}",
        ],
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for i in range(timeout, 0, -1):
        await asyncio.sleep(1)
        if process.poll() is not None:
            # already exit
            out, err = process.communicate()
            logger.warning(
                "subprocess exited, %s: %s", process.pid, out.decode("utf-8")
            )
            raise subprocess.SubprocessError(err.decode("utf-8"))
        if await is_local_omega_alive(port):
            logger.info("omega server is listen on %s", cfg.omega.urls.quotes_server)
            return process

    raise subprocess.SubprocessError("Omega server malfunction.")


def load_bars_from_file(name, ext: str = "csv", sep="\t"):
    file = os.path.join(data_dir(), f"{name}.{ext}")
    df = pd.read_csv(file, sep=sep, parse_dates=True)

    df["frame"] = pd.to_datetime(df["frame"])
    df.set_index("frame", inplace=True)
    dtype = [
        ("frame", "<M8[ms]"),
        ("open", "f4"),
        ("high", "f4"),
        ("low", "f4"),
        ("close", "f4"),
        ("volume", "f8"),
        ("amount", "f8"),
        ("factor", "f4"),
    ]

    return np.array(df.to_records(), dtype=dtype)
