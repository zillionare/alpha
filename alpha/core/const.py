from typing import Final

# 归类与回测相关的事件，比如回测开始、结束等
E_BACKTEST: Final = "BACKTEST"

# 归类与策略相关的事件，比如买入，卖出等。
E_STRATEGY: Final = "STRATEGY"

# 归类与远程执行策略相关的事件
## 启动远程回测
E_EXECUTOR_BACKTEST: Final = "ALPHA.EXECUTOR.BACKTEST"
## 当子进程初始化完成时，发出此消息
E_EXECUTOR_STARTED: Final = "ALPHA.EXECUTOR.STARTED"
## 通知子进程退出
E_EXECUTOR_EXIT: Final = "ALPHA.EXECUTOR.EXIT"
## 当子进程退出时，发出此消息，注意需要在emit.stop之前调用
E_EXECUTOR_EXITED: Final = "ALPHA.EXECUTOR.EXITED"
E_EXECUTOR_ERROR: Final = "ALPHA.EXECUTOR.ERROR"
# 发出方为父进程
E_EXECUTOR_ECHO_PARENT: Final = "ALPHA.EXECUTOR.ECHO.PARENT"
# 发出方为子进程
E_EXECUTOR_ECHO_CHILD: Final = "ALPHA.EXECUTOR.ECHO.CHILD"

# 常见股指
hs300: Final = "399300.XSHE"
