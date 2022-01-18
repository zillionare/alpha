from typing import List
import jqdatasdk as jq
import os
from jqdatasdk import finance, query
import pandas as pd
import numpy as np


class WildGooseStrategy:
    def __init__(self) -> None:
        self._authed = False

        if not self._authed:
            account = os.getenv("JQ_ACCOUNT")
            password = os.getenv("JQ_PASSWORD")
            jq.auth(account, password)
            self._authed = jq.is_auth()
        self.funds_info = None

    def rank_funds(self, cat: int, n: int = 10) -> list:
        """按曲线找出走势向上，波动最小的10支基金"""
        self.fetch_fund_list([cat])
        codes = self.get_code_list()
        results = self.fetch_net_values(codes, n)

        X = []
        scores = []
        codes = []
        for code, values in results.items():
            if len(values) < n or np.any(np.isnan(values.astype(float))):
                continue

            X.append(values)
            codes.append(code)

        X = np.array(X, dtype="<f4")

        # 低点以来的收益率
        matrix_min = np.min(X, axis=1)
        scores.append(X[:,-1]/matrix_min - 1)

        X = X[:,1:]/X[:,:-1] - 1
        # 波动率
        scores.append(np.std(X, axis=1))

        # 最大回撤
        scores.append(np.min(np.cumsum(X, axis=1), axis=1))

        # winrate
        scores.append(np.sum(X > 0, axis=1)/n)

        # 排名
        scores = pd.DataFrame(np.transpose(scores), index=codes, columns=["最大收益", "波动率", "最大回撤", "胜率"])

        scores["name"] = scores.index.map(self.code2name)

        df = scores[["name", "最大收益", "胜率", "最大回撤", "波动率"]].sort_values(by=["最大收益", "胜率", "最大回撤", "波动率"], ascending=[False, False, False, True])

        df["R收益"] = df["最大收益"].rank(ascending=False)
        df["R胜率"] = df["胜率"].rank(ascending=False)
        df["R回撤"] = df["最大回撤"].rank(ascending=False)
        df["R波动"] = df["波动率"].rank(ascending=False)

        return df

    def fetch_net_value(self, code:str, n:int) -> np.array:
        q=query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.code == code).order_by(finance.FUND_NET_VALUE.day.desc()).limit(n)

        df=finance.run_query(q)
        return df['sum_value'].values

    def fetch_net_values(self, codes: list, n:int) -> list:
        """获取基金`n`日的净值"""
        results = {}

        for code in codes:
            values = self.fetch_net_value(code, n)
            results[code] = values

        return results

    def fetch_fund_list(self, kind = [402001, 402004]):
        """获取基金列表"""
        q_ = query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.underlying_asset_type_id.in_(kind))

        dfs = []
        for i in range(100):
            q = q_.limit(3000).offset(i * 3000)
            df = finance.run_query(q)
            if len(df) > 0:
                dfs.append(df)
            else:
                break

        self.funds = pd.concat(dfs)

    def code2name(self, code:str)->str:
        """将基金代码转换为基金名称"""
        return self.funds[self.funds['main_code'] == code].iloc[0]["name"]


    def get_code_list(self)->List[str]:
        return self.funds['main_code'].tolist()


if __name__ == "__main__":
    s = WildGooseStrategy()
    print(s.rank_funds(402001, 20)[:20])

