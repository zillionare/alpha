from typing_extensions import OrderedDict
import numpy as np
from omicron.core.talib import cross
from alpha.core.features import long_parallel, moving_average, polyfit, short_parallel


class MaLineFeatures:
    def __init__(self, bars, wins=[5, 10, 20, 30, 60, 120], check_window=10):
        """均线特征，包括多（空）头排列，金叉、死叉, 一阳穿多线，一阴穿多线"""
        close = bars["close"]
        open_ = bars["open"]

        self.bars = bars
        self.wins = wins
        if wins == [5, 10, 20, 30, 60, 120]:
            self.parallel_groups = [(0, 1, 2), (3, 4, 5)]
        elif wins == [5, 10, 20, 30, 60]:
            self.parallel_groups = [(0, 1, 2), (2, 3, 4)]
        elif wins == [5, 10, 20, 30]:
            self.parallel_groups = [(0, 1, 2), (2, 3)]
        elif wins == [5, 10, 20]:
            self.parallel_groups = [(0, 1, 2), None]
        elif wins == [5, 10]:
            self.parallel_groups = [(0, 1), None]
        else:
            raise ValueError("invalid wins value")

        if len(bars) < max(self.wins) + check_window - 1:
            raise ValueError("not enough bars")

        self.cw = check_window

        mas = []
        features = {}
        for win in self.wins:
            ma = moving_average(close, win)
            if len(ma) < self.cw:
                return None

            mas.append(ma[-self.cw :].tolist())

        mas = np.array(mas, dtype="<f4")
        for pg in self.parallel_groups:
            if pg is None:
                continue

            wins = ",".join([f"{self.wins[i]}" for i in pg])
            key = f"parallel({wins})"
            flag, length = long_parallel(mas[pg, :])
            if flag:
                features[key] = length
            else:
                flag, length = short_parallel(mas)
                if flag:
                    features[key] = -1 * length
                else:
                    features[key] = 0

        # 金叉、死叉
        for i, (w1, w2) in enumerate(zip(self.wins[:-1], self.wins[1:])):
            key = f"{w1}x{w2}"
            flag, idx = cross(mas[i][-self.cw :], mas[i + 1][-self.cw :])
            idx = self.cw - idx
            features[key] = flag * idx

        # 支撑、阻力
        arr = [*mas[:, -1], close[-1]]
        arg_c = len(arr) - 1
        sorted_pos = np.argsort(arr)
        for i in range(len(sorted_pos)):
            if sorted_pos[i] == arg_c:
                if i == 0:
                    features["support"] = None
                    features["support_gap"] = None
                    pos = sorted_pos[i + 1]
                    features["supress"] = self.wins[pos]
                    features["supress_gap"] = round(close[-1] / arr[pos] - 1, 3)
                elif i == len(sorted_pos) - 1:
                    features["supress"] = None
                    features["supress_gap"] = None
                    pos = sorted_pos[i - 1]
                    features["support"] = self.wins[pos]
                    features["support_gap"] = round(close[-1] / arr[pos] - 1, 3)
                else:
                    pos = sorted_pos[i - 1]
                    features["support"] = self.wins[pos]
                    features["support_gap"] = round(close[-1] / arr[pos] - 1, 3)

                    pos = sorted_pos[i + 1]
                    features["supress"] = self.wins[pos]
                    features["supress_gap"] = round(close[-1] / arr[pos] - 1, 3)
                break

        # 一阳穿多线（或一阴穿多线）
        arr = [open_[-1], *mas[:, -1], close[-1]]
        arg_c = len(arr) - 1
        arg_o = 0
        sorted_pos = np.argsort(arr)
        wins = np.array(self.wins)

        # calc bull strike -> open < close
        start = np.argmax(sorted_pos == arg_o) + 1
        end = np.argmax(sorted_pos == arg_c)
        bull_strike = wins[sorted_pos[start:end] - 1]

        # calc bearish strike -> close < open
        start = np.argmax(sorted_pos == arg_c) + 1
        end = np.argmax(sorted_pos == arg_o)
        bearish_strike = wins[sorted_pos[start:end] - 1]

        features["bull_strike"] = bull_strike
        features["bearish_strike"] = bearish_strike

        # 参考趋势线
        pmae_threshold = 1e-3
        trendline = 0
        for i, ma in enumerate(mas):
            (a, b, c), pmae = polyfit(ma[-self.cw :] / ma[0])
            # 近乎直线，当前不破参考趋势线
            if pmae <= pmae_threshold and abs(a) < 1e-3:
                if b > 0 and np.all(close[-2:] > ma[-2:]):
                    trendline = self.wins[i]
                    break
                elif b < 0 and np.all(close[-2:] < ma[-2:]):
                    trendline = self.wins[i]
                    break

        features["trendline"] = trendline
        if trendline != 0:
            features["trendline_slope"] = b
        else:
            features["trendline_slope"] = None

        self.features = features
        self.feature_names = sorted(features.keys())

    def as_vector(self):
        vec = []
        for name in self.feature_names:
            if name.endswith("_strike"):
                vec.append(len(self.features[name]))
            else:
                vec.append(self.features[name])

        return vec

    def __str__(self) -> str:
        desc = []
        for pg in self.parallel_groups:
            if pg is None:
                continue

            wins = ",".join([f"{self.wins[i]}" for i in pg])
            key = f"parallel({wins})"

            parallelled_days = self.features[key]
            if parallelled_days > 0:
                desc.append(f"{'多头(' + wins + ')':<15}{parallelled_days:<15}")
            elif parallelled_days < 0:
                desc.append(f"{'空头(' + wins + ')':<15}{parallelled_days:<15}")
            else:
                pass

        for w1, w2 in zip(self.wins[:-1], self.wins[1:]):
            key = f"{w1}x{w2}"
            cross = self.features[key]
            if cross < 0:
                text = f"{abs(cross)}日前死叉"
                desc.append(f"{key:<16}{text:<15}")
            elif cross > 0:
                text = f"{abs(cross)}日前金叉"
                desc.append(f"{key:<16}{text:<15}")

        if self.features["support"] is not None:
            desc.append(f"{'支撑线':<14}{self.features['support']}日均线")
            desc.append(f"{'支撑位':<14}{self.features['support_gap']:.1%}")

        if self.features["supress"] is not None:
            desc.append(f"{'压力线':<14}{self.features['supress']}日均线")
            desc.append(f"{'压力位':<14}{self.features['supress_gap']:.1%}")

        slope = self.features["trendline_slope"]
        trendline = self.features["trendline"]
        if slope and slope > 0:
            text = f"沿{trendline}日线上行"
            desc.append(f"{'趋势线':<14}{text:<15}")
            desc.append(f"{'趋势线斜率':<14}{slope:.4f}")
        elif slope and slope < 0:
            text = f"沿{trendline}日线下行"
            desc.append(f"{'趋势线':<14}{text:<15}")
            desc.append(f"{'趋势线斜率':<14}{slope:.4f}")

        bull_strike = ",".join([str(x) for x in self.features["bull_strike"]])
        bearish_strike = ",".join([str(x) for x in self.features["bearish_strike"]])

        if bull_strike:
            desc.append(f"{'一阳穿多线':<13}{bull_strike}日均线")
        if bearish_strike:
            desc.append(f"{'一阴穿多线':<13}{bearish_strike}日均线")

        return "\n".join(desc)
