from types import FrameType
from typing import List, Tuple

import numpy as np
from coretypes import FrameType
from omicron.models.stock import Stock
from omicron.models.timeframe import TimeFrame
from omicron.talib import cross, moving_average, polyfit
from omicron.talib.common import find_runs, top_n_argpos

from alpha.core.features import parallel


class MaLineFeatures:
    def __init__(self, check_window=10):
        """均线特征，包括多（空）头排列，金叉、死叉, 一阳穿多线，一阴穿多线

        Args:
            bars (np.structured array): 日线数据,长度应大于wins[-1] + check_window - 1
        """
        self.cw = check_window
        common_columns = [
            ("trendline_slope", "{:.3f}"),
            ("trendline_gap", "{:.0f}"),
            ("trendline", "{:.0f}"),
            ("support", "{:.0f}"),
            ("support_gap", "{:.1%}"),
            ("supress", "{:.0f}"),
            ("supress_gap", "{:.1%}"),
            ("parallel", "{:.0f}"),
            ("follow_ma5", "{:.2f}"),
            ("dma5", "{:.2%}"),
            ("ddma5", "{:.2%}"),
            ("dma10", "{:.2%}"),
            ("ddma10", "{:.2%}"),
        ]

        # 5, 10 20, 30, 60, 120
        # 短期：10, 20, 30 ( 5, 10, 20 将被包含在全排列中)
        parallel_123 = ("parallel_123", "{:.0f}")
        # 中期：20, 30, 60
        parallel_234 = ("parallel_234", "{:.0f}")
        # 长期 30, 60, 120
        parallel_345 = ("parallel_345", "{:.0f}")

        line_params_20 = [
            ("a5", "{:.2%}"),
            ("b5", "{:.2%}"),
            ("pmae5", "{:.2%}"),
            ("vx5", "{:.1f}"),
            ("a10", "{:.2%}"),
            ("b10", "{:.2%}"),
            ("pmae10", "{:.2%}"),
            ("vx10", "{:.1f}"),
            ("a20", "{:.2%}"),
            ("b20", "{:.2%}"),
            ("pmae20", "{:.2%}"),
            ("vx20", "{:.1f}"),
        ]

        line_params_30 = [
            *line_params_20,
            ("a30", "{:.2%}"),
            ("b30", "{:.2%}"),
            ("pmae30", "{:.2%}"),
            ("vx30", "{:.1f}"),
        ]

        line_params_60 = [
            *line_params_30,
            ("a60", "{:.2%}"),
            ("b60", "{:.2%}"),
            ("pmae60", "{:.2%}"),
            ("vx60", "{:.1f}"),
        ]

        line_params_120 = [
            *line_params_60,
            ("a120", "{:.2%}"),
            ("b120", "{:.2%}"),
            ("pmae120", "{:.2%}"),
            ("vx120", "{:.1f}"),
        ]

        line_params_250 = [
            *line_params_120,
            ("a250", "{:.2%}"),
            ("b250", "{:.2%}"),
            ("pmae250", "{:.2%}"),
            ("vx250", "{:.1f}"),
        ]

        cross_columns_20 = [("cross_5x10", "{:.0f}"), ("cross_10x20", "{:.0f}")]

        cross_columns_30 = [*cross_columns_20, ("cross_20x30", "{:.0f}")]

        cross_columns_60 = [*cross_columns_30, ("cross_30x60", "{:.0f}")]

        cross_columns_120 = [*cross_columns_60, ("cross_60x120", "{:.0f}")]

        cross_columns_250 = [*cross_columns_120, ("cross_120x250", "{:.0f}")]

        bull_strike_columns_20 = [
            ("bull_strike_5", "{}"),
            ("bull_strike_10", "{}"),
            ("bull_strike_20", "{}"),
        ]
        bull_strike_columns_30 = [*bull_strike_columns_20, ("bull_strike_30", "{}")]

        bull_strike_columns_60 = [*bull_strike_columns_30, ("bull_strike_60", "{}")]

        bull_strike_columns_120 = [*bull_strike_columns_60, ("bull_strike_120", "{}")]

        bull_strike_columns_250 = [*bull_strike_columns_120, ("bull_strike_250", "{}")]

        bearish_strike_columns_20 = [
            ("bear_strike_5", "{}"),
            ("bear_strike_10", "{}"),
            ("bear_strike_20", "{}"),
        ]

        bearish_strike_columns_30 = [
            *bearish_strike_columns_20,
            ("bear_strike_30", "{}"),
        ]

        bearish_strike_columns_60 = [
            *bearish_strike_columns_30,
            ("bear_strike_60", "{}"),
        ]

        bearish_strike_columns_120 = [
            *bearish_strike_columns_60,
            ("bear_strike_120", "{}"),
        ]

        bearish_strike_columns_250 = [
            *bearish_strike_columns_120,
            ("bear_strike_250", "{}"),
        ]

        self.columns_20 = [
            *common_columns,
            *line_params_20,
            *cross_columns_20,
            *bull_strike_columns_20,
            *bearish_strike_columns_20,
        ]

        self.columns_30 = [
            *common_columns,
            *line_params_30,
            parallel_123,
            *cross_columns_30,
            *bull_strike_columns_30,
            *bearish_strike_columns_30,
        ]

        self.columns_60 = [
            *common_columns,
            *line_params_60,
            parallel_123,
            parallel_234,
            *cross_columns_60,
            *bull_strike_columns_60,
            *bearish_strike_columns_60,
        ]

        self.columns_120 = [
            *common_columns,
            *line_params_120,
            parallel_123,
            parallel_234,
            parallel_345,
            *cross_columns_120,
            *bull_strike_columns_120,
            *bearish_strike_columns_120,
        ]

        self.columns_250 = [
            *common_columns,
            *line_params_250,
            parallel_123,
            parallel_234,
            parallel_345,
            *cross_columns_250,
            *bull_strike_columns_250,
            *bearish_strike_columns_250,
        ]

    def common_feature(self, bars, mas, line_params):
        """共同的均线特征"""
        vector = []

        # vec(3): 趋势线斜率，当前股价与趋势线之间的距离，均线编号：-1表明不存在趋势线
        slope, gap, i = self.trend_line(bars, mas, line_params)
        vector.extend((slope, gap, i))

        # vec(2+2): 支撑线，压力线
        close = bars["close"]
        vector.extend(self.support_and_supress(close[-1], mas[:, -1]))

        # vec(1): 均线全排列情况。正数表明存在多头排列，负数表明存在空头排列，0表示不存在
        # 绝对值表示排列的天数
        vector.append(parallel(mas))

        # vec(1): 股价与5日均线之间的关系
        vector.append(self.follow_ma5(close, mas[0]))

        # 5日均线一阶导数和二阶导，可判断是上升拐头还是下降拐头
        ma5 = mas[0][-3:]
        dma5 = ma5[1:] / ma5[:-1] - 1
        ddma5 = dma5[1] - dma5[0]
        vector.extend((dma5[0], ddma5))

        # 10日均线的一阶导和二阶导，可判断是上升拐头还是下降拐头
        ma10 = mas[1][-3:]
        dma10 = ma10[1:] / ma10[:-1] - 1
        ddma10 = dma10[1] - dma10[0]
        vector.extend((dma10[0], ddma10))

        return vector

    def feature(self, bars, lines=4) -> np.array:
        if lines == 3:
            return self.feature_20(bars)
        elif lines == 4:
            return self.feature_30(bars)
        elif lines == 5:
            return self.feature_60(bars)
        elif lines == 6:
            return self.feature_120(bars)
        elif lines == 7:
            return self.feature_250(bars)
        else:
            raise ValueError("lines must be 3, 4, 5, 6, 7")

    def feature_30(self, bars) -> np.array:
        """提取[5,10,20, 30]均线周期的特征"""
        close = bars["close"]
        open_ = bars["open"]
        wins = [5, 10, 20, 30]
        vector = []

        if len(close) < (max(wins) + self.cw - 1):
            raise ValueError("bars length is too short")

        mas = self.mas(close, wins)
        line_params = self.line_params(mas, wins)

        vector.extend(self.common_feature(bars, mas, line_params))

        # 以下部分为可变长。长度由均线个数决定
        # vec(16): (a, b, pmae, vx) for 5, 10, 20, 30
        vector.extend(np.asarray(line_params).flatten())

        # vec(1): 短期均线(即10, 20, 30）排列情况
        vector.append(parallel(mas[1:4, :]))

        # vec(3): 5x10, 10x20, 20x30情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            vector.append(flag * idx)

        # vec(4): 一阳穿多线[1,0,0,0]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:, -1]))

        # vec(4): 一阴穿多线[0,0,1,1]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:, -1]))

        return vector

    def feature_60(self, bars) -> np.array:
        """提取[5,10,20,30, 60]均线周期的特征"""
        close = bars["close"]
        open_ = bars["open"]
        wins = [5, 10, 20, 30, 60]
        vector = []

        if len(close) < (max(wins) + self.cw - 1):
            raise ValueError("bars length is too short")

        mas = self.mas(close, wins)
        line_params = self.line_params(mas, wins)

        vector.extend(self.common_feature(bars, mas, line_params))

        # 以下部分为可变长。长度由均线个数决定
        # len(20): (a, b, pmae, vx) for 5, 10, 20, 30, 60
        vector.extend(np.asarray(line_params).flatten())

        # len(1) : 短期均线(即10, 20, 30）排列情况
        vector.append(parallel(mas[1:4, :]))

        # len(1) 中期均线（20， 30， 60）排列情况
        vector.append(parallel(mas[2:5, :]))

        # vec(27-29): 5x10, 10x20, 20x30, 30x60情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = int(self.cw - idx)
            vector.append(flag * idx)

        # len(5): 一阳穿多线[1,0,0,0,1]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:, -1]))

        # len(5): 一阴穿多线[0,0,1,1,0]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:, -1]))

        return vector

    def feature_120(self, bars):
        """提取[5, 10, 20, 30, 60, 120]均线周期的特征"""
        close = bars["close"]
        open_ = bars["open"]
        wins = [5, 10, 20, 30, 60, 120]
        vector = []

        if len(close) < (max(wins) + self.cw - 1):
            raise ValueError("bars length is too short")

        mas = self.mas(close, wins)
        line_params = self.line_params(mas, wins)

        vector.extend(self.common_feature(bars, mas, line_params))

        # 以下部分为可变长。长度由均线个数决定
        # vec(7-30): (a, b, pmae, vx) for 5, 10, 20, 30, 60, 120
        vector.extend(np.asarray(line_params).flatten())

        # 短期均线(10, 20, 30)排列情况
        vector.append(parallel(mas[1:4, :]))

        # 中期均线(20, 30, 60)排列情况
        vector.append(parallel(mas[2:5, :]))

        # 长期均线排列情况（30, 60, 120)
        vector.append(parallel(mas[3:6, :]))

        # vec(30-34): 5x10, 10x20, 20x30, 30x60, 60x120情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            vector.append(flag * idx)

        # vec(35-40): 一阳穿多线[1,0,0,0,1,0]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:, -1]))

        # vec(41-46): 一阴穿多线[0,0,1,1,0,0]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:, -1]))

        return vector

    def feature_250(self, bars):
        """提取[5, 10, 20, 30, 60, 120, 250]均线周期的特征"""
        close = bars["close"]
        open_ = bars["open"]
        wins = [5, 10, 20, 30, 60, 120, 250]
        vector = []

        if len(close) < (max(wins) + self.cw - 1):
            raise ValueError("bars length is too short")

        mas = self.mas(close, wins)
        line_params = self.line_params(mas, wins)

        vector.extend(self.common_feature(bars, mas, line_params))

        # 以下部分为可变长。长度由均线个数决定
        # vec(7-33): (a, b, pmae, vx) for 5, 10, 20, 30, 60, 120, 250
        vector.extend(np.asarray(line_params).flatten())

        # 短期均线(5, 10, 20)排列情况
        vector.append(parallel(mas[:3, :]))

        # 中期均线(20, 30, 60)排列情况
        vector.append(parallel(mas[2:5, :]))

        # 长期均线排列情况（30, 60, 120)
        vector.append(parallel(mas[3:6, :]))

        # vec(34-38): 5x10, 10x20, 20x30, 30x60, 60x120, 120x250情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            vector.append(flag * idx)

        # vec(39-44): 一阳穿多线[1,0,0,0,1,0,0]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:, -1]))

        # vec(45-50): 一阴穿多线[0,0,1,1,0,0,0]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:, -1]))

        return vector

    def line_params(self, mas, wins):
        line_params = []
        for i, ma in enumerate(mas):
            pl = self.polyfit_length(wins[i])
            ts = ma / ma[0]
            (a, b, _), pmae = polyfit(ts[-pl:])
            vx = pl - (-1 * b) / (2 * a)
            line_params.append((a, b, pmae, vx))
        return line_params

    def follow_ma5(self, close, ma5):
        """
        股价是否跟随ma5上行还是下行。在股价与均线的关系中，是否跟随5日均线有特别的意义。一般来说，股价并不会长期跟随更长期的均线运动。
        """
        # todo: the implementation is not correct
        close = close[-self.cw :]
        ma5 = ma5[-self.cw :]

        flags, _, length = find_runs(close >= ma5)
        if flags[-1]:
            return length[-1] / len(close)
        else:
            return -1 * length[-1] / len(close)

    def bull_strike(self, close, open_, mas) -> np.array:
        """提取一阳穿多线

        Args:
            close ([type]): [description]
            open_ ([type]): [description]
            mas ([type]): [description]

        Returns:
            np.array: 被穿过的均线序号，0-len(mas) - 1
        """
        arr = [open_, *mas, close]
        arg_c = len(arr) - 1
        arg_o = 0
        sorted_pos = np.argsort(arr)

        # calc bull strike -> open < close
        start = np.argmax(sorted_pos == arg_o) + 1
        end = np.argmax(sorted_pos == arg_c)
        striked = sorted_pos[start:end] - 1
        flags = np.zeros(len(mas), dtype=np.int8)
        flags[striked] = 1
        return flags.tolist()

    def bearish_strike(self, close, open_, mas) -> np.array:
        """提取一阴穿多线

        Args:
            close ([type]): [description]
            open_ ([type]): [description]
            mas ([type]): [description]

        Returns:
            np.array: 被穿过的均线序号，0-len(mas) - 1
        """
        arr = [open_, *mas, close]
        arg_c = len(arr) - 1
        arg_o = 0
        sorted_pos = np.argsort(arr)

        # calc bull strike -> open < close
        start = np.argmax(sorted_pos == arg_c) + 1
        end = np.argmax(sorted_pos == arg_o)
        striked = sorted_pos[start:end] - 1

        flags = np.zeros(len(mas), dtype=np.int8)
        flags[striked] = 1
        return flags.tolist()

    def support_and_supress(self, c, mas) -> Tuple[int, float, int, float]:
        """计算支撑位和压力位
        返回支撑的均线及距离，压力位的均线及距离。

        如果不存在支撑均线，则返回-1， 0.反之，如果不存在压力均线，则返回-1， 0
        """
        arr = [*mas, c]
        arg_c = len(arr) - 1
        sorted_pos = np.argsort(arr)
        support = -1
        support_gap = 0
        supress = -1
        supress_gap = 0

        i = sorted_pos.tolist().index(arg_c)
        if i == 0:  # close最小，只有压力位
            supress = sorted_pos[1]
            supress_gap = round(c / arr[supress] - 1, 3)
        elif i == len(arr) - 1:  # close最大，只有支撑位
            support = sorted_pos[-2]
            support_gap = round(c / arr[support] - 1, 3)
        else:
            support = sorted_pos[i - 1]
            supress = sorted_pos[i + 1]
            support_gap = round(c / arr[support] - 1, 3)
            supress_gap = round(c / arr[supress] - 1, 3)

        return int(support), support_gap, int(supress), supress_gap

    def mas(self, close, wins):
        mas = []
        n = len(close) - max(wins) + 1

        for w in wins:
            ma = moving_average(close, w)
            mas.append(ma[-n:])

        return np.asarray(mas)

    def trend_line(self, bars, mas, line_params, threshold=5e-3) -> Tuple[float, int]:
        """找出股价运行的趋势线。

        趋势线并不是支撑或者压力线，它是一条能大致代表股价运行方式的直线（均线）。我们从最短周期均线（5日均线）开始搜索，直到找到符合条件的均线，返回其斜率和均线序号。

        作为趋势线的均线，要求观测期间股价都不下破（如果是上升趋势线）该均线。反之亦然。

        趋势线不仅能反映股价运行的方向，还能反映股价变化的速度（涨速）

        Args:
            close ([type]): [description]
            mas ([type]): [description]
            line_params ([type]): [description]
            threshold ([type], optional): 可以接受的拟合误差。对指数或者30分钟线建议使用1e-3，对股票日线建议使用5e-3。 Defaults to 5e-3.

        Returns:
            Tuple[float, int]: 趋势线斜率和均线编号，-1表示不适用
        """
        bars = bars[-self.cw :]

        close = bars["close"]

        for i in range(len(line_params)):
            a, b, pmae, _ = line_params[i]
            if pmae >= threshold or abs(a) > threshold:
                continue

            ma = mas[i][-self.cw :]
            gap = close[-1] / ma[-1] - 1
            if b > 0:  # 上升直线
                # 不破均线
                if np.all(close >= ma):
                    return b, gap, i
            else:  # 下降
                # 不破均线
                if np.all(close <= ma):
                    return b, gap, i

        return 0, 0, -1

    def polyfit_length(self, w):
        return {5: 7, 10: 7}.get(w, 10)

    def explain(self, vec):
        """解释特征向量"""

        if len(vec) == len(self.columns_20):
            col_desc = self.columns_20
        elif len(vec) == len(self.columns_30):
            col_desc = self.columns_30
        elif len(vec) == len(self.columns_60):
            col_desc = self.columns_60
        elif len(vec) == len(self.columns_120):
            col_desc = self.columns_120
        elif len(vec) == len(self.columns_250):
            col_desc = self.columns_250
        else:
            raise ValueError("unknown vec length")

        desc = []
        for (col, fmt), v in zip(col_desc, vec):
            desc.append(f"{col}: {fmt.format(v)}")

        return desc

    def supress(self, bars, maline, torlerance=5e-3) -> bool:
        """检测bars是否被maline压制

        如果maline是下行（或水平）趋势线，且bars顺maline下行，返回True。即使收盘价高于均线，但只要是随均线下行，都认为是被压制。
        """
        # high代表了向上冲击maline
        high = bars["high"] * (1 + torlerance)
        low = bars["high"] * (1 - torlerance)

        n = min(len(bars), len(maline))

        bars = bars[-n:]
        maline = maline[-n:]

        half = n // 2
        if np.mean(maline[-half:]) >= np.mean(maline[: -half + 1]):
            # 均线非下行状态
            return False

        # 股价follows均线
        flags = (high >= maline) & (low <= maline)
        c0 = bars["close"][-1]

        return np.count_nonzero(flags) >= n * 0.85 and c0 <= maline[-1]

    def support(self, bars, maline, torlerance=5e-3) -> bool:
        """检测bars是否受maline支撑

        如果maline是上行（或水平）趋势线，且bars顺maline上行，返回True
        """
        # low代表了向下冲击maline
        low = bars["low"] * (1 - torlerance)
        high = bars["low"] * (1 + torlerance)

        n = min(len(bars), len(maline))

        bars = bars[-n:]
        maline = maline[-n:]

        half = n // 2
        if np.mean(maline[-half:]) <= np.mean(maline[: -half + 1]):
            # 均线非上行状态
            return False

        # 股价follows均线
        flags = (low <= maline) & (high >= maline)
        c0 = bars["close"][-1]

        return np.count_nonzero(flags) >= n * 0.85 and c0 >= maline[-1]

    def to_dict(self, vec):
        if len(vec) == len(self.columns_20):
            col_desc = self.columns_20
        elif len(vec) == len(self.columns_30):
            col_desc = self.columns_30
        elif len(vec) == len(self.columns_60):
            col_desc = self.columns_60
        elif len(vec) == len(self.columns_120):
            col_desc = self.columns_120
        elif len(vec) == len(self.columns_250):
            col_desc = self.columns_250
        else:
            raise ValueError("unknown vec length")

        d = {}
        for (col, _), v in zip(col_desc, vec):
            d[col] = v

        return d
