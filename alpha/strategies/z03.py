"""z03是一个基于均线和成交量特征的xgboost模型。

我们首先探索5、10、20、30、60、120、250日均线的模型。
"""

import pickle
import random
from omicron.models.security import Security
from omicron.models.securities import Securities
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from scipy.stats import randint, uniform
import arrow
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from alpha.core.features import fillna, moving_average
from sklearn.metrics import mean_squared_error as MSE


class Z03:
    def __init__(self, model_path: str = None):
        if model_path is not None:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = None

    def train(
        self, learning_rate, n_estimators, max_depth, subsample, colsample_bytree, gamma
    ):
        model = XGBRegressor(
            "reg:squarederror",
            learning_rate=learning_rate,
            n_estimators=n_estimators,  # 树的个数-10棵树建立xgboost
            max_depth=max_depth,  # 树的深度
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            seed=123,
        )

        (X_train, y_train), (X_valid, y_valid) = self.load_train_data(
            "/tmp/z03_train.pkl"
        )
        model.fit(X_train, y_train["y"])

        y_pred = model.predict(X_valid)
        rmse = np.sqrt(MSE(y_valid["y"], y_pred))
        for i in range(10):
            print(y_pred[i], y_valid[i])
        print("rmse of the model is", rmse)

        return model

    def predict(self, bars):
        if hasattr(bars, "close"):
            close = bars["close"]
            vol = bars["vol"]
            close = fillna(close)
            close = fillna(vol)

            X = self.transform(close, vol)
            c = bars["close"][-1]
        else:
            X = bars.reshape(1, -1)
            c = None

        pred = self.model.predict(X)
        if c:
            return c * pred
        else:
            return pred

    async def extract_features(
        self, ft: FrameType = FrameType.DAY, total=1000, ts_len=10, result_win=5
    ):
        """从k线数据中提取特征"""
        if isinstance(ft, str):
            ft = FrameType(ft)

        X = []
        y = []

        n = 250 + ts_len + result_win

        j = 0
        secs = Securities()

        # 生成长度略大于total的随机股票代码列表
        codes = secs.choose(_types=["stock"])
        codes = random.sample(codes * int(total / len(codes) + 1), int(total * 1.2))
        for code in codes:
            sec = Security(code)

            # 在过去一年中，随机取一段，以使得分布不依赖于时间
            stop = tf.shift(arrow.now(), -random.randint(result_win, 250), ft)
            start = tf.shift(stop, -n, ft)

            try:
                bars = await sec.load_bars(start, stop, ft)
                if len(bars) < n:
                    continue
                close = fillna(bars["close"])
                vol = fillna(bars["volume"])
                feature, (y_, c) = self.transform(close, vol, ts_len, result_win)
                X.append(feature)
                y.append((y_, code, start, c))
            except Exception:
                continue

            j += 1
            if j >= total:
                break

            if (j) % 100 == 0:
                print(f"{j}/{total} data recs generated")

        return {
            "X": np.array(X),
            "y": np.array(
                y,
                dtype=[
                    ("y", "<f4"),
                    ("sec", "O"),
                    ("start", "O"),
                    ("c", "<f4"),
                ],
            ),
        }

    def transform(self, close, vol, ts_len=10, res_win=0) -> np.array:
        """返回提取的特征（70列）及对应的目标值，如果res_win不为0


        Args:
            close ([type]): [description]
            vol ([type]): [description]
            ts_len (int, optional): [description]. Defaults to 10.
            res_win (int, optional): [description]. Defaults to 0.

        Returns:
            np.array: [description]
        """
        if res_win == 0:
            close_for_train = close
            volume_for_train = vol
        else:
            close_for_train = close[:-res_win]
            volume_for_train = vol[:-res_win]

        c = close_for_train[-1]

        close_for_train = close_for_train / c
        volume_for_train = volume_for_train[1:] / volume_for_train[:-1]

        ma5 = moving_average(close_for_train, 5)[-ts_len:]
        # ma10 = moving_average(close_for_train, 10)[-ts_len:]
        # ma20 = moving_average(close_for_train, 20)[-ts_len:]
        # ma30 = moving_average(close_for_train, 30)[-ts_len:]
        # ma60 = moving_average(close_for_train, 60)[-ts_len:]
        # ma120 = moving_average(close_for_train, 120)[-ts_len:]
        # ma250 = moving_average(close_for_train, 250)[-ts_len:]

        features = [
            *ma5,
            # *ma10,
            # *ma20,
            # *ma30,
            # *ma60,
            # *ma120,
            # *ma250,
            *volume_for_train[-ts_len:],
        ]

        if res_win != 0:
            y = close[-res_win] / c
        else:
            c, y = None, None, None

        return features, (y, c)

    def load_train_data(self, data_file: str, valid_pct=0.2):
        """加载训练用的数据。

        加载训练用的数据，进行shuffle, split
        """
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        X = data["X"]
        y = data["y"]

        # do the suffle
        np.random.seed(78)
        indice = np.random.choice(len(X), size=len(X), replace=False)
        X = X[indice]
        y = y[indice]

        valid_len = int(len(X) * valid_pct)
        train_len = len(X) - valid_len

        return (X[:train_len], y[:train_len]), (X[train_len:], y[train_len:])

    def grid_search(self):
        params = {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 1),
            "max_depth": randint(2, 6),
            "n_estimators": randint(100, 150),
            "subsample": uniform(0.6, 0.4),
        }
        model = XGBRegressor()
        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=42,
            n_iter=200,
            cv=3,
            verbose=1,
            n_jobs=1,
            return_train_score=True,
        )

        (X_train, y_train), (X_valid, y_valid) = self.load_train_data(
            "/tmp/z03_train.pkl"
        )

        search.fit(X_train, y_train["y"])

        self._report_best_scores(search.cv_results_)

    def _report_best_scores(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_score"] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results["mean_test_score"][candidate],
                        results["std_test_score"][candidate],
                    )
                )
                print("Parameters: {0}".format(results["params"][candidate]))
                print("")
