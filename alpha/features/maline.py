from typing import List, Tuple
import numpy as np
from omicron.core.talib import cross
from alpha.core.features import moving_average, parallel, polyfit


class MaLineFeatures:
    def __init__(self, check_window=10):
        """均线特征，包括多（空）头排列，金叉、死叉, 一阳穿多线，一阴穿多线

        Args:
            bars (np.structured array): 日线数据,长度应大于wins[-1] + check_window - 1
        """
        self.cw = check_window
        common_columns = [
            "trendline_slope",
            "trendline",
            "support",
            "support_gap",
            "supress",
            "supress_gap",
            "parallel",
        ]

        line_params_20 = [
            "a5",
            "b5",
            "pmae5",
            "vx5",
            "a10",
            "b10",
            "pmae10",
            "vx10",
            "a20",
            "b20",
            "pmae20",
            "vx20",
        ]

        line_params_30 = [
            *line_params_20,
            "a30",
            "b30",
            "pmae30",
            "vx30"]

        line_params_60 = [
            *line_params_30,
            "a60",
            "b60",
            "pmae60",
            "vx60"
        ]

        line_params_120 = [
            *line_params_60,
            "a120",
            "b120",
            "pmae120",
            "vx120"
        ]

        line_params_250 = [
            *line_params_120,
            "a250",
            "b250",
            "pmae250",
            "vx250"
        ]

        cross_columns_20 = [
            "cross_5x10",
            "cross_10x20",
        ]
        cross_columns_30 = [
            *cross_columns_20,
            "cross_20x30"
        ]

        cross_columns_60 = [
            *cross_columns_30,
            "cross_30x60",
        ]

        cross_columns_120 = [
            *cross_columns_60,
            "cross_60x120",
        ]

        cross_columns_250 = [
            *cross_columns_120,
            "cross_120x250",
        ]

        bull_strike_columns_20 = [
            "bull_strike_5",
            "bull_strike_10",
            "bull_strike_20",
        ]
        bull_strike_columns_30 = [
            *bull_strike_columns_20,
            "bull_strike_30"
        ]

        bull_strike_columns_60 = [
            *bull_strike_columns_30,
            "bull_strike_60",
        ]

        bull_strike_columns_120 = [
            *bull_strike_columns_60,
            "bull_strike_120",
        ]

        bull_strike_columns_250 = [
            *bull_strike_columns_120,
            "bull_strike_250",
        ]

        bearish_strike_columns_20 = [
            "bear_strike_5",
            "bear_strike_10",
            "bear_strike_20",
        ]
        bearish_strike_columns_30 = [
            *bearish_strike_columns_20,
            "bear_strike_30"
        ]

        bearish_strike_columns_60 = [
            *bearish_strike_columns_30,
            "bear_strike_60",
        ]

        bearish_strike_columns_120 = [
            *bearish_strike_columns_60,
            "bear_strike_120",
        ]

        bearish_strike_columns_250 = [
            *bearish_strike_columns_120,
            "bear_strike_250",
        ]

        self.columns_20 = [
            *common_columns,
            *line_params_20,
            *cross_columns_20,
            *bull_strike_columns_20,
            *bearish_strike_columns_20
        ]

        self.columns_30 = [
            *common_columns,
            *line_params_30,
            *cross_columns_30,
            *bull_strike_columns_30,
            *bearish_strike_columns_30
        ]

        self.columns_60 = [
            *common_columns,
            *line_params_60,
            *cross_columns_60,
            *bull_strike_columns_60,
            *bearish_strike_columns_60
        ]

        self.columns_120 = [
            *common_columns,
            *line_params_120,
            *cross_columns_120,
            *bull_strike_columns_120,
            *bearish_strike_columns_120
        ]

        self.columns_250 = [
            *common_columns,
            *line_params_250,
            *cross_columns_250,
            *bull_strike_columns_250,
            *bearish_strike_columns_250
        ]

    def common_feature(self, bars, mas, line_params):
        """共同的均线特征"""
        vector = []

        # vec(0-1): 趋势线斜率，均线编号：-1表明不存在趋势线
        slope, i = self.trend_line(bars, mas, line_params)
        vector.extend((slope, i))

        # vec(2-5): 支撑线，压力线
        close = bars["close"][-1]
        vector.extend(self.support_and_supress(close, mas[:,-1]))

        # vec(6): 均线全排列情况。正数表明存在多头排列，负数表明存在空头排列，0表示不存在
        # 绝对值表示排列的天数
        vector.append(parallel(mas))

        return vector

    def feature_30(self, bars) -> np.array:
        """提取[5,10,30]均线周期的特征"""
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
        # vec(7-22): (a, b, pmae, vx) for 5, 10, 20, 30
        vector.extend(np.asarray(line_params).flatten())

        # vec(23~25): 5x10, 10x20, 20x30情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            vector.append(flag * idx)

        # vec(26~29): 一阳穿多线[1,0,0,0]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:,-1]))

        # vec(30~33): 一阴穿多线[0,0,1,1]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:,-1]))

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
        # vec(7-26): (a, b, pmae, vx) for 5, 10, 20, 30, 60
        vector.extend(np.asarray(line_params).flatten())

        # vec(26-29): 5x10, 10x20, 20x30, 30x60情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            vector.append(flag * idx)

        # vec(30-34): 一阳穿多线[1,0,0,0,1]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:,-1]))

        # vec(35-39): 一阴穿多线[0,0,1,1,0]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:,-1]))

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

        # vec(30-34): 5x10, 10x20, 20x30, 30x60, 60x120情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            vector.append(flag * idx)

        # vec(35-40): 一阳穿多线[1,0,0,0,1,0]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:,-1]))

        # vec(41-46): 一阴穿多线[0,0,1,1,0,0]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:,-1]))

        return vector

    def feature_250(self, bars):
        """ 提取[5, 10, 20, 30, 60, 120, 250]均线周期的特征"""
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

        # vec(34-38): 5x10, 10x20, 20x30, 30x60, 60x120, 120x250情况
        for i in range(len(wins) - 1):
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            vector.append(flag * idx)

        # vec(39-44): 一阳穿多线[1,0,0,0,1,0,0]
        vector.extend(self.bull_strike(close[-1], open_[-1], mas[:,-1]))

        # vec(45-50): 一阴穿多线[0,0,1,1,0,0,0]
        vector.extend(self.bearish_strike(close[-1], open_[-1], mas[:,-1]))

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

        return support, support_gap, supress, supress_gap

    def mas(self, close, wins):
        mas = []
        n = len(close) - max(wins) + 1

        for w in wins:
            ma = moving_average(close, w)
            mas.append(ma[-n:])

        return np.asarray(mas)

    def trend_line(
        self, bars, mas, line_params, threshold=5e-3) -> Tuple[float, int]:
        """找出股价运行的趋势线。

        趋势线并不是支撑或者压力线，它是一条能大致代表股价运行方式的直线（均线）。我们从最短周期均线（5日均线）开始搜索，直到找到符合条件的均线，返回其斜率和均线序号。

        作为趋势线的均线，要求难测期间股价都不下破（如果是上升趋势线）该均线。反之亦然。

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
            if b > 0: # 上升直线
                # 不破均线
                if np.all(close >= ma):
                    return b, i
            else: # 下降
                # 不破均线
                if np.all(close <= ma):
                    return b, i

        return 0, -1

    def polyfit_length(self, w):
        return {5: 7, 10: 7}.get(w, 10)

    def explain(self, vec):
        """解释特征向量"""
        if len(vec) == 27:
            return {k:v for k,v in zip(self.columns_20, vec)}
        elif len(vec) == 34:
            return {k:v for k,v in zip(self.columns_30, vec)}
        elif len(vec) == 41:
            return {k:v for k,v in zip(self.columns_60, vec)}
        elif len(vec) == 48:
            return {k:v for k,v in zip(self.columns_120, vec)}
        elif len(vec) == 55:
            return {k:v for k,v in zip(self.columns_250, vec)}
        else:
            raise ValueError("unknown vec length")
