#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
import os
import signal
import subprocess
import sys
import time
import pprint

import aiohttp
import cfg4py
import fire
import psutil
from omicron.core.lang import async_run
from sanic import Sanic
from termcolor import colored

from alpha.config import get_config_dir

logger = logging.getLogger(__name__)

app = Sanic("alpha")
cfg = cfg4py.get_instance()


def find_alpha_process():
    for p in psutil.process_iter():
        cmd = " ".join(p.cmdline())
        if "alpha.app" in cmd:
            return p.pid

    return None


def start():
    print(f"正在启动{colored('zillionare-alpha', 'green')}...")
    init()

    proc = find_alpha_process()
    if proc is not None:
        print("zillionare-alpha is already started")
        sys.exit(0)

    subprocess.Popen([sys.executable, '-m', 'alpha.app', 'start'])


def stop():
    proc = find_alpha_process()
    if proc is None:
        print("zillionare-alpha is not started.")
    else:
        try:
            os.kill(proc, signal.SIGTERM)
            time.sleep(1)
        except Exception:
            pass


def restart():
    init()
    stop()
    start()


@async_run
async def scan(plot_name: str, **params):
    init()
    params['plot'] = plot_name
    params['cmd'] = 'scan'
    url = f"{cfg.alpha.urls.service}/plot"
    async with aiohttp.ClientSession() as client:
        try:
            async with client.post(url, json=params) as resp:
                if resp.status != 200:
                    print(colored('failed to execute scan', 'red'))
                else:
                    print(await resp.json())
        except Exception as e:
            print(e)


@async_run
async def monitor(cmd, *args, **kwargs):
    init()

    url = f"{cfg.alpha.urls.service}/monitor/{cmd}"

    async with aiohttp.ClientSession() as client:
        try:
            async with client.post(url, json=kwargs) as resp:
                if resp.status != 200:
                    print(colored('failed to execute scan', 'red'))
                else:
                    result = await resp.json()
                    if isinstance(result, list):
                        for item in result:
                            print(item)
                    else:
                        print(result)
        except Exception as e:
            print(e)


def init():
    server_roles = ['PRODUCTION', 'TEST', 'DEV']
    if os.environ.get(cfg4py.envar) not in ['PRODUCTION', 'TEST', 'DEV']:
        print(f"请设置环境变量{colored(cfg4py.envar, 'red')}为["
              f"{colored(server_roles, 'red')}]之一。")
        sys.exit(-1)

    config_dir = get_config_dir()
    cfg4py.init(config_dir, False)


def main():
    fire.Fire({
        "start":   start,
        "restart": restart,
        "stop":    stop,
        "scan":    scan,
        "monitor": monitor
    })


if __name__ == "__main__":
    main()
