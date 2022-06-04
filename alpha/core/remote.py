"""execute backtest and etc in remote process
"""

import asyncio
import logging
import os
import pickle
import socket
import threading
import time
from contextlib import closing
from multiprocessing import Process

import dill as pickle
import rpyc
from rpyc.utils.server import ThreadedServer

logger = logging.getLogger(__name__)


class RemoteService(rpyc.Service):
    """构建在rpyc之上的远程服务。

    本类实现功能：
    1. 创建远程服务进程池。如果未主动创建，则会在调用`remote_call`方法时，自动创建一个默认进程池。
    2. 在远程服务进程中，创建了一个专属线程以运行asyncio loop，以便客户端可以提交执行远程的异步代码。注意，即使是提交的异步代码，调用者也是阻塞式地等待其返回结果。
    3. 对外只暴露一个`remote_call`方法，用于在远程服务进程中执行异步代码。
    4. 如果需要远程执行的方法运行时，需要有初始化和终结化动作，可以向RemoteService添加相应的监听器，
    ```
        service = RemoteService()
        service.add_event_listener("on_connect", init)
        service.add_event_listener("on_disconnect", close)
    ```
    5. 一组service使用一个RemoteService对象，但可以有多个执行进程。
    """

    def __init__(self, timeout=30, on_connect=None, on_disconnect=None):
        self.loop = asyncio.new_event_loop()

        # thread for executing asyncio loop
        self.thread = None

        self.peer = None
        self.ports = []
        self.conns = []
        self.cur = 0
        self.instance = None
        self.listeners = {"on_connect": on_connect, "on_disconnect": on_disconnect}

        self.timeout = timeout

        # timeout for initialization/termination
        self.timeout = timeout

    def _run_coroutine(self, awaitable):
        return asyncio.run_coroutine_threadsafe(awaitable, loop=self.loop)

    def on_connect(self, conn):
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()

        self.peer = conn

        on_connect = self.listeners.get("on_connect")
        if on_connect:
            if asyncio.iscoroutinefunction(on_connect):
                future = self._run_coroutine(on_connect())

                # wait until the future is done
                _ = future.result(timeout=self.timeout)
            else:
                self.on_connect()

        return super().on_connect(conn)

    def on_disconnect(self, conn):
        on_disconnect = self.listeners.get("on_disconnect")

        if on_disconnect:
            if asyncio.iscoroutinefunction(on_disconnect):
                future = self._run_coroutine(on_disconnect())

                # wait until the future is done
                _ = future.result(timeout=self.timeout)
            else:
                self.on_disconnect()

        return super().on_disconnect(conn)

    def exposed_echo(self, msg):
        return msg

    def exposed_call(self, byte_code, args=None, kwargs=None):
        """this is where magic happens"""
        func = pickle.loads(byte_code)

        if args is not None:
            args = pickle.loads(args)
        else:
            args = ()

        if kwargs is not None:
            kwargs = pickle.loads(kwargs)
        else:
            kwargs = {}

        logger.debug("remote_call is called with args %s and kwargs %s", args, kwargs)
        if asyncio.iscoroutinefunction(func):
            future = self._run_coroutine(func(*args, **kwargs))
            logger.debug("type of future in %s is %s", os.getpid(), type(future))
            return future

        return func(*args, **kwargs)

    def remote_entry(self, port: int):
        """远程服务入口函数

        Args:
            port : 监听端口
        """
        server = ThreadedServer(
            self,
            port=port,
            protocol_config={"allow_public_attrs": True, "sync_request_timeout": None},
        )
        server.start()

    def find_free_port(self):
        """查找空闲端口"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def create_service_pool(self, workers: int = 1, timeout: int = 10):
        """创建远程服务进程池

        Args:
            workers : 进程池中进程数量

        Raises:
            RuntimeError: 如果远程服务无法正常响应echo方法，则抛出此异常
        """
        self.ports = [self.find_free_port() for _ in range(workers)]

        processes = []
        for port in self.ports:
            proc = Process(target=self.remote_entry, args=(port,))
            proc.start()
            logger.info("start process(%s) listen at %s", proc.pid, port)
            processes.append(proc)

            for i in range(0, timeout):
                try:
                    conn = rpyc.connect(
                        "localhost",
                        port,
                        config={
                            "allow_public_attrs": True,
                            "sync_request_timeout": None,
                        },
                        keepalive=True,
                    )
                    if conn.root.echo(7080) == 7080:
                        self.conns.append(conn)
                        break
                except ConnectionRefusedError:
                    pass
                except Exception as e:
                    logger.exception(e)

                time.sleep(1)
            else:
                raise RuntimeError(f"remote service at {port} is not available")

    def remote_call(self, func, *args, **kwargs):
        """远程调用函数

        func既可以是一个普通函数，也可以是awaitable对象。远程服务会判断func类型并且正确调用。如果是awaitable对象，则会将其转换为future对象，并且返回future对象。即此时的调用为异步调用（但不是coroutine）。

        注意在进行远程调用时，如果func依赖了一些非builtin的模块，需要在`func`中进行导入。
        Args:
            func : 远程调用的函数
            args : 函数的参数
            kwargs : 函数的关键字参数
        """
        if len(self.conns) == 0:
            self.create_service_pool()

        self.cur = (self.cur + 1) % len(self.conns)
        conn = self.conns[self.cur]

        byte_code = pickle.dumps(func)

        if len(args) > 0:
            args = pickle.dumps(args)
        else:
            args = None

        if len(kwargs):
            kwargs = pickle.dumps(kwargs)
        else:
            kwargs = None

        res = conn.root.call(byte_code, args=args, kwargs=kwargs)

        return res


__all__ = ["remote_call", "RemoteService"]
