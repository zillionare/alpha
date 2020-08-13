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

import cfg4py
import fire
import psutil
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
    server_roles = ['PRODUCTION', 'TEST', 'DEV']
    if os.environ.get(cfg4py.envar) not in ['PRODUCTION', 'TEST', 'DEV']:
        print(f"请设置环境变量{colored(cfg4py.envar, 'red')}为["
              f"{colored(server_roles, 'red')}]之一。")
        sys.exit(-1)

    config_dir = get_config_dir()
    cfg4py.init(config_dir, False)

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
    stop()
    start()


def main():
    fire.Fire({
        "start":   start,
        "restart": restart,
        "stop":    stop
    })


if __name__ == "__main__":
    main()
