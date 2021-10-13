"""月线数据来训练一个模型，判断是否可以用均线来预测股价"""

import jqdatasdk as jq
import os
import arrow
import datetime
import numpy as np
from jqdatasdk import *
import pandas as pd

account = os.getenv("JQ_ACCOUNT")
password = os.getenv("JQ_PASSWORD")

jq.auth(account, password)
valuations = jq.get_fundamentals(query(valuation))
secs = jq.get_all_securities()


def get_name(code):
    return secs[secs.index == code].iloc[0]["display_name"]


def get_valuation(code):
    return int(valuations[valuations.code == code].iloc[0]["market_cap"])


def choose_stocks(exclude_st=True, exclude_688=True, valuation_range=(100, 2000)):
    result = secs
    if exclude_688:
        result = result[result.index.str.startswith("688") == False]
    if exclude_st:
        result = result[result.display_name.str.find("ST") == -1]

    codes = []
    for code in result.index:
        try:
            if valuation_range[0] <= get_valuation(code) <= valuation_range[1]:
                codes.append(code)
        except Exception:
            pass

    return codes


def top_n_argpos(ts: np.array, n: int) -> np.array:
    """get top n (max->min) elements and return argpos which its value ordered in descent

    Example:
        >>> top_n_argpos([4, 3, 9, 8, 5, 2, 1, 0, 6, 7], 2)
        array([2, 3])
    Args:
        ts (np.array): [description]
        n (int): [description]

    Returns:
        np.array: [description]
    """
    return np.argsort(ts)[-n:][::-1]


def moving_average(ts: np.array, win: int):
    """计算时间序列ts在win窗口内的移动平均

    Example:

        >>> ts = np.arange(7)
        >>> moving_average(ts, 5)
        >>> array([2.0000, 3.0000, 4.0000])

    """

    return np.convolve(ts, np.ones(win) / win, "valid")


def get_bars(code, n, end, unit="1d"):
    fields = ["date", "open", "high", "low", "close", "volume"]
    return jq.get_bars(
        code,
        n,
        unit=unit,
        end_dt=end,
        fq_ref_date=end,
        df=False,
        include_now=True,
        fields=fields,
    )


def predict_by_moving_average(
    ts, win: int, n_preds: int = 1, err_threshold=1e-2, n: int = None
) -> float:
    """predict the next ith value by fitted moving average

    make sure ts is not too long and not too short

    Args:
        ts (np.array): the time series
        i (int): the index of the value to be predicted, start from 1
        win (int): the window size
        n (int): how many ma sample points used to polyfit the ma line

    Returns:
        float: the predicted value
    """
    ma = moving_average(ts, win)

    # how many ma values used to fit the trendline?
    if n is None:
        n = {5: 7, 10: 10}.get(win, 15)

    if len(ma) < n:
        raise ValueError(f"{len(ma)} < {n}, can't predict")

    coef, pmae = polyfit(ma[-n:], degree=2)
    if pmae > err_threshold:
        return None, None

    # build the trendline with same length as ma
    fitma = np.polyval(coef, np.arange(n - len(ma), n + n_preds))

    preds = [
        reverse_moving_average(fitma[: i + 1], i, win)
        for i in range(len(ma), len(ma) + n_preds)
    ]

    return preds, pmae


def polyfit(ts: np.array, degree: int = 2) -> tuple:
    """fit ts with np.polyfit, return coeff and pmae"""
    coeff = np.polyfit(np.arange(len(ts)), ts, degree)
    pmae = np.abs(np.polyval(coeff, np.arange(len(ts))) - ts).mean() / np.mean(ts)
    return coeff.tolist(), pmae


def reverse_moving_average(ma, i: int, win: int) -> float:
    """given moving_average, reverse the origin value at index i

    if i < win, then return Nan, these values are not in the window thus cannot be recovered

    see also https://stackoverflow.com/questions/52456267/how-to-do-a-reverse-moving-average-in-pandas-rolling-mean-operation-on-pr
    but these func doesn't perfom well with out moving_average
    Example:
        >>> c = np.arange(10)
        >>> ma = moving_average(c, 3)
        >>> c1 = [reverse_moving_average(ma, i, 3) for i in range(len(ma))]
        >>> c1 == [1, 2, 3, 4.9999, 6.000, 7, 8, 9]

    Args:
        ma (np.array): the moving average series
        i (int): the index of origin
        win (int): the window size, which is used to calculate moving average
    """
    return ma[i] * win - ma[i - 1] * win + np.mean(ma[i - win : i])


import pickle


def make_ds_month(path):
    end = arrow.get("2021-08-31").date()
    data = {}
    for code in secs[secs.start_date < datetime.datetime(2018, 1, 1, 0, 0, 0)].index:
        bars = get_bars(code, 68, end, unit="1M")
        data[code] = bars

    path = os.path.expanduser(path)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=5)


def load_monthly_bars():
    with open("/Users/aaronyang/data/monthly_bars.pkl", "rb") as f:
        return pickle.load(f)


def search():
    results = []
    pred_samples = {5: 7, 10: 10, 20: 10}
    for code, bars in load_monthly_bars().items():
        for i in range(29, len(bars) - 1):
            bars_ = bars[i - 29 : i]

            frame = bars_["date"][-1]
            c0 = bars_["close"][-1]
            c1 = bars[i]["close"]
            pcr = c1 / c0 - 1

            row = [code, get_name(code), frame, pcr]
            for win in [5, 10, 20]:
                ma = moving_average(bars_["close"], win)[-10:]
                ma /= ma[0]
                (a, b, c), pmae = polyfit(ma)

                ypreds, _ = predict_by_moving_average(
                    bars_["close"], win, 1, 1, pred_samples[win]
                )
                if ypreds is not None:
                    pred = ypreds[0] / c0 - 1
                else:
                    pred = None

                row.extend((a, b, pmae, pred))

            results.append(row)

    return pd.DataFrame(
        results,
        columns=[
            "code",
            "name",
            "frame",
            "actual",
            "a5",
            "b5",
            "pmae5",
            "pred5",
            "a10",
            "b10",
            "pmae10",
            "pred10",
            "a20",
            "b20",
            "pmae20",
            "pred20",
        ],
    )


def preprocess(df):
    """prepare dataset for xgboost classification

    0 不可预测
    1 可用5月线预测
    2 可用10月线
    3 可用20月线
    """
    return {
        "X": df[
            [
                "a5",
                "b5",
                "pmae5",
                "pred5",
                "a10",
                "b10",
                "pmae10",
                "pred10",
                "a20",
                "b20",
                "pmae20",
                "pred20",
            ]
        ].values,
        "y": labelling(df),
    }


def labelling(df, threshold=0.1):
    """"""
    labels = []
    for i in range(len(df)):
        pred5 = df.loc[i, "pred5"]
        pred10 = df.loc[i, "pred10"]
        pred20 = df.loc[i, "pred20"]
        actual = df.loc[i, "actual"]

        d = np.abs(np.array([pred5 - actual, pred10 - actual, pred20 - actual]))
        if min(d) > threshold:
            labels.append(0)
            continue

        pos = np.argmin(d)
        labels.append(pos + 1)

    return labels


from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform


def train(X_train, y_train, X_test, y_test):
    model = XGBClassifier()

    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.01, 1),
        "max_depth": randint(2, 6),
        "n_estimators": randint(80, 150),
        "subsample": uniform(0.6, 0.4),
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        random_state=78,
        n_iter=50,
        cv=3,
        n_jobs=8,
        return_train_score=True,
        refit=True,  # do the refit oursel
    )

    fit_params = {
        "eval_set": [(X_test, y_test)],
        "early_stopping_rounds": 5,
    }

    try:
        search.fit(X_train, y_train, **fit_params)
    except Exception as e:
        print(e)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    report = classification_report(y_test, preds)
    print(report)
    return best_model


if __name__ == "__main__":
    # df = search()
    # df.to_pickle("/tmp/month.df")

    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # df = pd.read_pickle("/tmp/month.df")

    # data = preprocess(df)
    # with open("/tmp/month.ds", "wb") as f:
    #     pickle.dump(data, f)

    with open("/tmp/month.ds", "rb") as f:
        data = pickle.load(f)

    X, y = shuffle(data["X"], data["y"], random_state=78)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=78
    )
    model = train(X_train, y_train, X_test, y_test)

    save_to = os.path.expanduser("~/data/models/month_ma.pkl")
    with open(save_to, "wb") as f:
        pickle.dump(model, f)
