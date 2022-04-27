"""execute backtest and etc in remote process
"""

import asyncio
import logging
import pickle
import socket
import threading
import time
from contextlib import closing
from multiprocessing import Process

import dill as pickle
import rpyc
from rpyc.utils.server import ThreadedServer
import os

logger = logging.getLogger(__name__)


class RemoteService(rpyc.Service):
    """构建在rpyc之上的远程服务。

    本类实现功能：
    1. 创建远程服务进程池。如果未主动创建，则会在调用`remote_call`方法时，自动创建一个默认进程池。
    2. 在远程服务进程中，创建了一个专属线程以运行asyncio loop，以便客户端可以提交执行远程的异步代码。注意，即使是提交的异步代码，调用者也是阻塞式地等待其返回结果。
    3. 对外只暴露一个`remote_call`方法，用于在远程服务进程中执行异步代码。
    """

    ports = []
    conns = []
    cur = 0
    instance = None
    listeners = {}

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        self.peer = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls, *args, **kwargs)

        return cls.instance

    @classmethod
    def add_event_listener(cls, name: str, func):
        assert name in (
            "on_connect",
            "on_disconnect",
        ), f"{name} is not a valid event name"

        cls.listeners[name] = func

    def _submit_async(self, awaitable):
        return asyncio.run_coroutine_threadsafe(awaitable, loop=self.loop)

    def on_connect(self, conn):
        self.peer = conn

        on_connect = self.listeners.get("on_connect")
        if on_connect:
            if asyncio.iscoroutinefunction(on_connect):
                future = self._submit_async(on_connect())

                # wait until the future is done
                _ = future.result(timeout=10)
            else:
                on_connect()

        return super().on_connect(conn)

    def on_disconnect(self, conn):
        on_disconnect = self.listeners.get("on_disconnect")

        if on_disconnect:
            if asyncio.iscoroutinefunction(on_disconnect):
                future = self._submit_async(on_disconnect())

                # wait until the future is done
                _ = future.result(timeout=10)
            else:
                on_disconnect()

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
            future = self._submit_async(func(*args, **kwargs))
            logger.info("type of future in %s is %s", os.getpid(), type(future))
            return future.result()

        return func(*args, **kwargs)

    @classmethod
    def remote_entry(cls, port: int):
        """远程服务入口函数

        Args:
            port : 监听端口
        """
        if cls.instance is None:
            cls.instance = RemoteService()
        server = ThreadedServer(
            cls.instance,
            port=port,
            protocol_config={"allow_public_attrs": True, "sync_request_timeout": None},
        )
        server.start()

    @classmethod
    def find_free_port(cls):
        """查找空闲端口"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @classmethod
    def create_service_pool(cls, workers: int = 1, timeout: int = 10):
        """创建远程服务进程池

        Args:
            workers : 进程池中进程数量

        Raises:
            RuntimeError: 如果远程服务无法正常响应echo方法，则抛出此异常
        """
        cls.ports = [cls.find_free_port() for _ in range(workers)]

        processes = []
        for port in cls.ports:
            proc = Process(target=cls.remote_entry, args=(port,))
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
                        cls.conns.append(conn)
                        break
                except ConnectionRefusedError:
                    pass
                except Exception as e:
                    logger.exception(e)

                time.sleep(1)
            else:
                raise RuntimeError(f"remote service at {port} is not available")

    @classmethod
    def remote_call(cls, func, *args, **kwargs):
        """远程调用函数

        func既可以是一个普通函数，也可以是awaitable对象。远程服务会判断func类型并且正确调用。

        Args:
            func : 远程调用的函数
            args : 函数的参数
            kwargs : 函数的关键字参数
        """
        if len(cls.conns) == 0:
            cls.create_service_pool()

        cls.cur = (cls.cur + 1) % len(cls.conns)
        conn = cls.conns[cls.cur]

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


remote_call = RemoteService.remote_call

__all__ = ["remote_call", "RemoteService"]
