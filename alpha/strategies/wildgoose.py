"""寻找基金中走势平稳向上的
"""
import datetime
import logging
import os
from enum import Enum, IntEnum
from typing import List

import h5py
import jqdatasdk as jq
import numpy as np
import pandas as pd
from coretypes import FrameType
from jqdatasdk import finance, query
from omicron.models.timeframe import TimeFrame as tf
from empyrical import max_drawdown
from omicron.talib.morph import polyfit

logger = logging.getLogger(__name__)


class OperateMode(IntEnum):
    OF = 401001
    CF = 401002
    QDII = 401003
    FOF = 401004
    ETF = 401005
    LOF = 401006
    MOM = 401007
    RETIS = 401008

    @property
    def name(self):
        name_map = {
            401001: "开放式基金",
            401002: "封闭式基金",
            401003: "QDII",
            401004: "FOF",
            401005: "ETF",
            401006: "LOF",
            401007: "MOM",
            401008: "基础设施基金",
        }


class AssetType(IntEnum):
    STOCK = 402001
    CURRENCY = 402002
    BOND = 402003
    MIXED = 402004
    FUND = 402005
    METAL = 402006
    CLOSED = 402007

    @property
    def name(self):
        name_map = {
            402001: "股票",
            402002: "货币",
            402003: "债券",
            402004: "混合",
            402005: "基金",
            402006: "贵金属",
            402007: "封闭式",
        }
        return name_map[self.value]


class InvestStyle(Enum):
    STOCK_ONLY = "005001"
    STOCK_INDEX = "005003"
    BALANCED = "005004"
    STOCK_FIRST = "005005"
    BOND_FIRST = "005006"
    CAPITAL_FIRST = "005007"

    # ST: short term
    ST_BOND = "005008"
    BOND = "005009"
    L1_BOND = "005010"
    L2_BOND = "005011"

    BOND_INDEX = "005012"
    FUND = "005013"
    CURRENCY = "005014"

    ST_WEALTH = "005015"
    GOLD = "005016"
    OTHER = "005017"

    @property
    def name(self):
        name_map = {
            "005005": "偏股混合型",
            "005011": "二级债基",
            "005009": "长期纯债型",
            "005013": "基金型",
            "005014": "货币型",
            "005010": "一级债基",
            "005012": "债券指数型",
            "005008": "中短期纯债型",
            "005004": "平衡混合型",
            "005001": "普通股票型",
            "005003": "增强指数型",
            "005006": "偏债混合型",
            "005015": "短期理财债券型",
            "005007": "保本型",
            "005016": "贵金属商品",
            "005017": "其他商品",
        }
        return name_map[str(self.value)]


class Target:
    DOMESTIC_STOCK_PURE = 0
    DOMESTIC_STOCK_FIRST = 1


class WildGooseStrategy:
    def __init__(self) -> None:
        self._authed = False

        if not self._authed:
            account = os.getenv("JQ_ACCOUNT")
            password = os.getenv("JQ_PASSWORD")
            jq.auth(account, password)
            self._authed = jq.is_auth()

        self.fetch_fund_list(None)

    def choose(
        self,
        modes: List[OperateMode] = None,
        asset_types: List[AssetType] = None,
        styles: List[InvestStyle] = None,
        end_dt: datetime.date = None,
    ) -> List[str]:
        """选择基金"""
        end_dt = end_dt or datetime.date.today()
        funds = self.funds[
            (
                self.funds["end_date"].isna()
                | (~self.funds["end_date"].isna() & (self.funds["end_date"] > end_dt))
            )
            & (self.funds["start_date"] < end_dt)
        ]

        if modes is not None:
            funds = funds[funds["operate_mode_id"].isin(modes)]

        if asset_types is not None:
            funds = funds[funds["underlying_asset_type_id"].isin(asset_types)]

        if styles is not None:
            styles = [style.value for style in styles]
            funds = funds[funds["invest_style_id"].isin(styles)]

        return funds.main_code.tolist()

    def adjust(self, X: np.array) -> np.array:
        row, col = X.shape

        # roc = X[:,1:]/X[:,:-1] - 1

        w = [(i + col) / col for i in range(col)]
        # coef = np.hstack([np.ones((row, 1)), roc * w + 1])

        # return np.cumprod(coef, axis=1)
        return X * w

    def rank_funds(self, target_type: Target, n: int = 10, n_fund=-1) -> list:
        """按曲线找出走势向上，波动最小的10支基金"""
        if target_type == Target.DOMESTIC_STOCK_PURE:
            modes = [OperateMode.OF, OperateMode.CF, OperateMode.ETF, OperateMode.LOF]
            invest_styles = [InvestStyle.STOCK_INDEX, InvestStyle.STOCK_ONLY]
        elif target_type == Target.DOMESTIC_STOCK_FIRST:
            modes = [OperateMode.OF, OperateMode.CF, OperateMode.ETF, OperateMode.LOF]
            invest_styles = [InvestStyle.STOCK_FIRST]

        codes = self.choose(modes=modes, styles=invest_styles)

        if n_fund == -1:
            n_fund = len(codes)

        codes = codes[:n_fund]
        results = self.fetch_net_values(codes, n)

        X = []
        codes = []
        for code, values in results.items():
            if len(values) < n or np.any(np.isnan(values.astype(float))):
                continue

            X.append(values)
            codes.append(code)

        X = np.array(X, dtype="<f4")

        # 计算斜率。为了使后面的变化更加重要，对数据进行了加权处理
        SLP = []
        PMAE = []
        MDD = []

        # X_ = self.adjust(X)
        for x in X:
            (slp, _), pmae = polyfit(x, degree=1)
            SLP.append(slp)
            PMAE.append(pmae)
            try:
                mdd, _, _ = max_drawdown(x)
                MDD.append(mdd)
            except:
                MDD.append(0)

        # 低点以来的收益率
        matrix_min = np.min(X, axis=1)
        PROFIT = X[:, -1] / matrix_min - 1

        # 以下数据使用ROC来计算
        ROC = X[:, 1:] / X[:, :-1] - 1

        # 胜率
        WR = np.sum(ROC > 0, axis=1) / n

        # 波动率
        VOLATILITY = np.std(ROC, axis=1)

        data = np.array(
            [
                [self.code2name(code) for code in codes],
                WR,
                MDD,
                SLP,
                PMAE,
                VOLATILITY,
                PROFIT,
            ]
        ).T
        df = pd.DataFrame(
            data,
            index=codes,
            columns=["基金名称", "胜率", "最大回撤", "斜率", "拟合误差", "波动率", "收益率"],
        )

        # 过滤掉走势不向上的基金，或者收益率 < 2%的基金
        df = df[(df["斜率"].astype(float) > 0) & (df["收益率"].astype(float) > 0.02)]

        # 排名
        rank_fields = ["胜率", "最大回撤", "斜率", "波动率"]

        return df.sort_values(by=rank_fields, ascending=[False, True, False, True])

    def fetch_net_value(self, code: str, n: int) -> np.array:
        q = (
            query(finance.FUND_NET_VALUE)
            .filter(finance.FUND_NET_VALUE.code == code)
            .order_by(finance.FUND_NET_VALUE.day.desc())
            .limit(n)
        )

        df = finance.run_query(q)
        return df["sum_value"].values[::-1]

    def fetch_net_values(self, codes: list, n: int) -> list:
        """获取基金`n`日的净值"""
        results = {}

        for code in codes:
            values = self.fetch_net_value(code, n)
            results[code] = values

        return results

    def fetch_fund_list(self, kind=[402001, 402004]):
        """获取基金列表"""
        if kind:
            q_ = query(finance.FUND_MAIN_INFO).filter(
                finance.FUND_MAIN_INFO.underlying_asset_type_id.in_(kind)
            )
        else:
            q_ = query(finance.FUND_MAIN_INFO)

        dfs = []
        for i in range(100):
            q = q_.limit(3000).offset(i * 3000)
            df = finance.run_query(q)
            if len(df) > 0:
                dfs.append(df)
            else:
                break

        funds = pd.concat(dfs)
        self.funds = funds[funds.end_date.isna()]

    def code2name(self, code: str) -> str:
        """将基金代码转换为基金名称"""
        return self.funds[self.funds["main_code"] == code].iloc[0]["name"]

    def plot(self, code: str, n: int = 20) -> None:
        """[summary]

        Args:
            code (str): [description]
            n (int, optional): [description]. Defaults to 20.
        """
        from matplotlib import pyplot as plt

        values = self.fetch_net_value(code, n)
        plt.plot(values)
        plt.text(
            0.01,
            0.9,
            f"{code}: {self.code2name(code)}",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        plt.show()


if __name__ == "__main__":
    s = WildGooseStrategy()
    s.run()
