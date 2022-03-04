"""市场环境概览
"""
from typing import Dict, List, Tuple
from coretypes import FrameType
from omicron.models.stock import Stock
from omicron.models.timeframe import TimeFrame
from alpha.core.features import *
from alpha.core.rsi_stats import RsiStats
from omicron.core.triggers import FrameTrigger
import arrow
from alpha.features.maline import MaLineFeatures
from alpha.core.rsi_stats import rsi30, rsiday
import numpy as np

from alpha.features.volume import moving_net_volume, top_volume_direction


class Glance:
    common_cols = [
        ("roc", "{:.1%}"),
        ("altitude", "{:.0%}"),
        ("rsi_5", "{:.1f}"),
        ("rsi_4", "{:.1f}"),
        ("rsi_3", "{:.1f}"),
        ("rsi_2", "{:.1f}"),
        ("rsi_1", "{:.1f}"),
        ("up", "{:.1%}"),
        ("down", "{:.1%}"),
        ("body", "{:.1%}"),
        ("up_shadow", "{}"),
        ("down_shadow", "{}"),
        ("double_top", "{}"),
        ("double_bottom", "{}"),
        ("dark_cloud_cover", "{}"),
        ("piercing_line", "{}"),
        ("three_crows", "{}"),
        ("three_red_soldiers", "{}"),
        ("top1_vol", "{:.1f}"),
        ("reverse_vol", "{:.1f}"),
        ("net_vol", "{:.1f}"),
        ("magic_number", "{}"),
    ]

    def __init__(self):
        self.rsi_day = RsiStats("ris", FrameType.DAY)
        self.rsi_30m = RsiStats("ris", FrameType.MIN30)

        self.rsi_day.load()
        self.rsi_30m.load()

        self.xshg = Stock("000001.XSHG")
        self.glance = []

    def register_backend_jobs(self, scheduler):
        trigger = FrameTrigger(FrameType.MIN30, "1s")
        scheduler.add_job(self.update_stats, trigger, args=(FrameType.MIN30,))

        trigger = FrameTrigger(FrameType.DAY, "1h")
        scheduler.add_job(self.update_stats, trigger, args=(FrameType.DAY,))

    async def update_stats(self, frame_type):
        if frame_type == FrameType.MIN30:
            # always calc for 1000 frames
            end = TimeFrame.floor(arrow.now(), frame_type)
            start = TimeFrame.shift(end, -1000, frame_type)

            if not (
                start == self.rsi_30m.time_range[0]
                and end == self.rsi_30m.time_range[1]
            ):
                await self.rsi_30m.calc(frame_type)
        elif frame_type == FrameType.DAY:
            # always calc for 1000 frames
            end = TimeFrame.floor(arrow.now(), frame_type)
            start = TimeFrame.shift(end, -1000, frame_type)

            if not (
                start == self.rsi_day.time_range[0]
                and end == self.rsi_day.time_range[1]
            ):
                await self.rsi_day.calc(frame_type)

    async def update_status(self):
        """报告当前状态"""
        status = {}

        # to check if reach RSI top/bottom
        end = TimeFrame.floor(arrow.now(), FrameType.MIN30)
        start = TimeFrame.shift(end, -40, FrameType.MIN30)

        shbars = await self.xshg.load_bars(start, end, FrameType.MIN30)
        close = shbars["close"]
        rsi = relative_strength_index(close, 6)
        p = self.rsi_30m.get_proba(self.xshg.code, rsi)

        status["rsi"] = rsi
        status["prsi"] = p

    @classmethod
    def diagnose(
        cls,
        code,
        bars,
        ft: FrameType,
        print_to_console=False,
        pmae_thres=3e-3,
        a5_thres=3e-3,
    ) -> Tuple[bool, List, Dict]:
        """诊断股票

        Args:
            code ([type]): [description]
            bars ([type]): [description]
            ft (FrameType): [description]
            pmae_thres ([type], optional): [description]. Defaults to 3e-3.
            a5_thres ([type], optional): [description]. Defaults to 3e-3.
        """

        sec = Stock(code)
        name = sec.display_name
        close = bars["close"]
        roc = close[-1] / close[-2] - 1

        vec = []

        # roc
        vec.append(roc)

        # Altitude
        vec.append(altitude(bars))

        # VOLUME
        vec.extend(top_volume_direction(bars))
        vec.append(moving_net_volume(bars)[-1])

        # rsi
        rsi = relative_strength_index(close, 6)
        vec.extend(rsi[-5:])

        # 形态特征
        vec.extend(morph_patterns(bars))

        # 是否处于整数关口
        vec.append(magic_number(bars[-1]))

        # 均线特征
        mf = MaLineFeatures()

        values = {}

        for i, (k, _) in enumerate([*cls.common_cols, *mf.columns_30]):
            values[k] = vec[i]

        bull = []
        bear = []

        bull_single_col = []
        bear_single_col = []

        others = []

        # ROC
        roc = values["roc"]
        if roc > 0:
            bull.append(("上涨中", f"{roc:.2%}"))
        elif roc < 0:
            bear.append(("下跌中", f"{roc:.2%}"))

        # ALTITUDE
        altitude = values["altitude"]
        if altitude >= 0.89:
            others.append("阶段高位")
        elif altitude <= 0.11:
            others.append("阶段低位")
        else:
            others.append(f"海拔[{altitude:.1f}]")

        # RSI
        stats = rsi30 if ft == FrameType.MIN30 else rsiday
        rsi = np.array([values[k] for k in ("rsi_3", "rsi_2", "rsi_1")])
        prsi = np.array([stats.get_proba(code, r) for r in rsi])

        if np.all(prsi):
            if any(prsi > 0.9):
                if prsi[-1] > 0.9:
                    bear.append(("RSI当前高位", f"{prsi[-1]:.1%}"))
                else:
                    bear.append(("RSI近期高位", f"{max(prsi):.1%}"))
            if any(prsi < 0.1):
                if prsi[-1] < 0.1:
                    bull.append(("RSI当前低位", f"{1 - prsi[-1]:.1%}"))
                else:
                    bull.append((f"RSI近期低位", f"{1 - min(prsi):.1%}"))
        else:
            if any(rsi > 85):
                if rsi[-1] > 90:
                    bear.append(("RSI当前高位", f"{rsi[-1]:.1f}"))
                else:
                    bear.append(("RSI近期高位", f"{max(rsi):.1f}"))
            if any(rsi < 15):
                if rsi[-1] < 15:
                    bull.append(("RSI当前低位", f"{1 - rsi[-1]:.1f}"))
                else:
                    bull.append(("RSI近期低位", f"{1 - min(rsi):.1f}"))

        # SHADOW
        up = values["up"]
        is_long = values["up_shadow"]
        if is_long:
            bear.append(("长上影线", f"{up:.1%}"))
        down = values["down"]
        is_long = values["down_shadow"]
        if is_long:
            bull.append(("长下影线", f"{down:.1%}"))

        # 单列
        # DOUBLE TOP/BOTTOM
        if values["double_top"] > 0 and altitude > 0.89:
            bear_single_col.append("双顶")
        if values["double_bottom"] > 0 and altitude < 0.11:
            bull_single_col.append("双底")

        # DARK CLOUD CORVER
        if values["dark_cloud_cover"]:
            bear_single_col.append("高开大阴线")

        # 低开大阳
        if values["piercing_line"]:
            bull_single_col.append("低开大阳线")

        # three crows
        if values["three_crows"]:
            bear_single_col.append("三只乌鸦")

        # three red soldiers
        if values["three_red_soldiers"]:
            bull_single_col.append("红三兵")

        # 大资金方向
        vol1 = values["top1_vol"]
        vol2 = values["reverse_vol"]
        if vol1 >= 3.5 and vol2 > -0.5:
            bull.append(("放量涨缩量跌", f"{vol1:.1f}/{vol2:.1f}"))
        elif vol1 > 2.5:
            bull.append(("温和放量上涨", f"{vol1:.1f}/{vol2:.1f}"))
        elif vol1 <= -3.5 and vol2 < 0.5:
            bear.append(("放量跌缩量涨", f"{vol1:.1f}/{vol2:.1f}"))
        elif vol1 < -2.5:
            bull.append(("温和放量下跌", f"{vol1:.1f}/{vol2:.1f}"))
        else:
            others.append(f"大资金力度{vol1:.1f}， 反向力度{vol2:.1f}")

        # 净余量
        net_vol = values["net_vol"]
        if net_vol > 0:
            bull.append(("资金流入", f"{net_vol:.1f}"))
        elif net_vol < 0:
            bear.append(("资金流出", f"{net_vol:.1f}"))

        # 整数关口
        if values["magic_number"]:
            if altitude >= 0.89:
                bear_single_col.append("整数关口压力")
            elif altitude <= 0.11:
                bull_single_col.append("整数关口支撑")
            else:
                others.append("整数关口")

        # 均线特征
        ## 趋势线
        trendline = values["trendline"]
        wins = [5, 10, 20, 30, 60, 120, 250]
        if trendline != -1:
            slope = values["trendline_slope"]
            if slope > 0:
                bull.append((f"顺{wins[trendline]}上升", f"{slope:.1%}"))
            elif slope < 0:
                bear.append((f"顺{wins[trendline]}下降", f"{slope:.1%}"))

        ## 支撑位、压力位
        support = values["support"]
        if support != -1:
            gap = values["support_gap"]
            others.append(f"支撑线{wins[support]}, 距离{gap:.1%}")

        supress = values["supress"]
        if supress != -1:
            gap = values["supress_gap"]
            others.append(f"压力线{wins[supress]}, 距离{gap:.1%}")

        ## 多头、空头排列
        parallel = values["parallel"]
        if parallel > 0:
            bull.append(("多头排列", f"{parallel}"))
        elif parallel < 0:
            bear.append(("空头排列", f"{-parallel}"))

        ## 5日均线伴随
        rel5 = values["ma5_relation"]
        if rel5 >= 0.4:
            bull.append(("均线之上", f"{rel5:.1%}"))
        elif rel5 > 0:
            others.append(f"均线之上 {rel5:.1%}")
        elif rel5 <= -0.4:
            bear.append(("均线之下", f"{rel5:.1%}"))
        elif rel5 < 0:
            others.append(f"均线之下 {-rel5:.1%}")

        ## 均线动量
        dma5_1 = values["dma5_1"]
        dma5_2 = values["dma5_2"]
        if dma5_1 > 0 and dma5_2 < 0:
            bear.append(("动量下拐(5)", f"{dma5_1:.2%}/{dma5_2:.2%}"))
        elif dma5_1 < 0 and dma5_2 > 0:
            bull.append(("动量上拐(5)", f"{dma5_1:.2%}/{dma5_2:.2%}"))
        elif dma5_1 > 0 and dma5_2 > 0:
            bull.append(("动量上行(5)", f"{dma5_2:.1%}/{dma5_2:.2%}"))
        elif dma5_1 < 0 and dma5_2 < 0:
            bear.append(("动量下行(5)", f"{dma5_1:.2%}/{dma5_2:.2%}"))

        dma10_1 = values["dma10_1"]
        dma10_2 = values["dma10_2"]
        if dma10_1 > 0 and dma10_2 < 0:
            bear.append(("动量下拐(10)", f"{dma10_1:.2%}/{dma10_2:.2%}"))
        elif dma10_1 < 0 and dma10_2 > 0:
            bull.append(("动量上拐(10)", f"{dma10_1:.2%}/{dma10_2:.2%}"))
        elif dma10_1 > 0 and dma10_2 > 0:
            bull.append(("动量上行(10)", f"{dma10_2:.1%}/{dma10_2:.2%}"))
        elif dma10_1 < 0 and dma10_2 < 0:
            bear.append(("动量下行(10)", f"{dma10_1:.2%}/{dma10_2:.2%}"))

        ## 均线趋势
        for win in [wins]:
            key = f"pame{win}"
            pmae_n = values.get(key)
            if not pmae_n or pmae_n > pmae_thres:
                continue

            a = values.get(f"a{win}")
            vx = values.get(f"vx{win}")
            if a > a5_thres and vx > 2:
                bull.append((f"{win}日均线加速上升", f"{a:.1%}/{vx:.0f}"))
            elif a < -a5_thres and vx > 2:
                bear.append((f"{win}日均线加速下降", f"{a:.1%}/{vx:.0f}"))

        ## 金叉死叉
        for k in ["cross_5x10", "cross_10x20", "cross_20x30"]:
            cross = values.get(k)
            name = k.replace("cross_", "")
            if cross > 0:
                bull.append((f"金叉({name})", f"{cross}"))
            elif cross < 0:
                bear.append((f"死叉({name})", f"{-cross}"))

        ## 上穿、跌破均线
        bull_strike = []
        bear_strike = []
        for win in [5, 10, 20, 30, 60, 120, 250]:
            k = f"bull_strike_{win}"
            if values.get(k) == 1:
                bull_strike.append(f"{win}")
            k = f"bear_strike_{win}"
            if values.get(k) == 1:
                bear_strike.append(f"{win}")

        if len(bull_strike) > 0:
            bull_strike = " ".join(bull_strike)
            bull.append(("上破均线", bull_strike))
        if len(bear_strike) > 0:
            bear_strike = " ".join(bear_strike)
            bear.append(("跌破均线", bear_strike))

        output = []
        output.append("======== 看多信号 ========")
        for k, v in bull:
            output.append(f"{k:<16}{v}")

        for msg in bull_single_col:
            output.append(msg)

        output.append("\n======== 看空信号 ========")
        for k, v in bear:
            output.append(f"{k:<16}{v}")

        for msg in bear_single_col:
            output.append(msg)

        output.append("\n======== 一般信号 ========")
        output.extend(others)

        if print_to_console:
            print("\n".join(output))

        return (
            len(bull) + len(bull_single_col) > len(bear) + len(bear_single_col),
            output,
            values,
        )
