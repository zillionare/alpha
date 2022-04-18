"""大盘30分钟顶底预测
"""
import os
import pickle
import random

import arrow
import numpy as np
import psutil
from coretypes import FrameType
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from alpha.core.features import (
    altitude,
    dark_cloud_cover,
    double_bottom,
    double_top,
    peaks_and_valleys,
    piercing_line,
    relative_strength_index,
    shadow_features,
    three_crows,
    three_red_soldiers,
)
from alpha.features.maline import MaLineFeatures
from alpha.features.volume import moving_net_volume, top_volume_direction
from alpha.notebook import get_bars


class IndexShPeakValleys:
    def __init__(self, version, inference_mode=False):
        self.ver = version
        self.code = "000001.XSHG"
        self.data_path = f"~/zillionare/alpha/data/pv/{self.ver}"
        self.data_path = os.path.expanduser(self.data_path)

        self.model_path = f"~/zillionare/alpha/models/pv/{self.ver}"
        self.model_path = os.path.expanduser(self.model_path)

        if inference_mode:
            file = os.path.join(self.model_path, "sh30m.pv.xgb.pkl")
            with open(file, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = None

        self.common_cols = [
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

    def features(self, bars, malines=4):
        """
        1. 当前altitude
        2. RSI(最后5日）
        4. 当前点上影、实体、下影线比例
        5. 当前点的涨跌幅
        6. 量能特征
        7. 5日内有没有新高、新低？
        8. 是否处于整数关口？
        8. 均线特征
        """
        vector = []
        close = bars["close"]

        # 当期涨跌幅 roc
        vector.append(close[-1] / close[-2] - 1)

        # altitude
        vector.append(altitude(bars))

        # rsi
        rsi = relative_strength_index(close)
        vector.extend(rsi[-5:])

        # 形态特征
        up, down, body = shadow_features(bars[-1])
        long_up_shadow = abs(up) > abs(body) and abs(up) > 0.03
        long_down_shadow = abs(down) > abs(body) and abs(down) > 0.03

        ## shadow
        vector.extend((up, down, body))
        vector.extend((long_up_shadow, long_down_shadow))

        ## 双顶？双底
        vector.append(double_top(bars))
        vector.append(double_bottom(bars))

        ## 乌云盖顶
        vector.append(dark_cloud_cover(bars[-2:]))

        ## 刺穿线
        vector.append(piercing_line(bars[-2:]))

        ## 三支乌鸦
        vector.append(three_crows(bars))

        ## 红三兵
        vector.append(three_red_soldiers(bars))

        # 量能特征
        ## 成交量方向
        vector.extend(top_volume_direction(bars))
        ## mobv: 净余买入量
        vector.append(moving_net_volume(bars)[-1])

        # 是否处于整数关口
        vector.append(self.magic_number(bars[-1]))

        # 均线特征，变长
        mf = MaLineFeatures()
        if malines == 3:
            vector.extend(mf.feature_20(bars))
        elif malines == 4:
            vector.extend(mf.feature_30(bars))
        elif malines == 5:
            vector.extend(mf.feature_60(bars))
        elif malines == 6:
            vector.extend(mf.feature_120(bars))
        elif malines == 7:
            vector.extend(mf.feature_250(bars))

        return vector

    def explain(self, vec):
        """
        说明模型
        """
        mf = MaLineFeatures()
        if len(vec) == len(self.common_cols) + len(mf.columns_30):
            col_desc = [*self.common_cols, *mf.columns_30]

        desc = []
        for (col, fmt), v in zip(col_desc, vec):
            desc.append(f"{col}: {fmt.format(v)}")

        return desc

    async def make_dataset(self, save_to=None):
        bars = await get_bars(self.code, 12000, "30m")
        peaks, valleys = peaks_and_valleys(bars["close"])

        features = []
        for i in peaks:
            if i < 100 or i == len(bars) - 2:
                continue

            # 对每一个顶和底，我们都使用当前点及其后一个点的数据来进行学习
            # 因为有一些特征，可能是当前点就能预测的，有一些特征，则需要等待信号确认的时间
            for j in [i, i + 1]:
                bars_ = bars[j - 99 : j + 1]
                feature, _ = reversal_features(self.code, bars_, FrameType.MIN30)
                features.append([*feature, 0, j])

        for i in valleys:
            if i < 100 or i == len(bars) - 1:
                continue

            for j in [i, i + 1]:
                bars_ = bars[j - 99 : j + 1]
                feature, _ = reversal_features(self.code, bars_, FrameType.MIN30)
                features.append([*feature, 1, j])

        # 将顶、底及前后各一个点的数据排除掉，这些数据容易干扰顶底的预测
        peaks = np.asarray(peaks)
        pv = set(peaks)
        pv.update(set(peaks - 1))
        pv.update(set(peaks + 1))

        valleys = np.asarray(valleys)
        pv.update(set(valleys))
        pv.update(set(valleys - 1))
        pv.update(set(valleys + 1))

        middle = list(set(np.arange(100, len(bars) - 2)).difference(pv))
        for i in middle:
            if i < 100 or i == len(bars) - 2:
                continue

            bars_ = bars[i - 99 : i + 1]
            feature, _ = reversal_features(self.code, bars_, FrameType.MIN30)
            features.append([*feature, 2, i])

        features = np.array(features)

        # split train and test
        total = len(features)

        train_indice = random.sample(list(np.arange(total)), int(total * 0.9))
        test_indice = list(set(np.arange(total)) - set(train_indice))

        X_train = features[train_indice][:, :-2]
        y_train = features[train_indice][:, -2].astype("i4")
        meta_train = features[train_indice][:, -1].astype("i4")

        X_test = features[test_indice][:, :-2]
        y_test = features[test_indice][:, -2].astype("i4")
        meta_test = features[test_indice][:, -1].astype("i4")

        data = {
            "bars": bars,
            "peaks": peaks,
            "valleys": valleys,
            "X_train": X_train,
            "y_train": y_train,
            "meta_train": meta_train,
            "X_test": X_test,
            "y_test": y_test,
            "meta_test": meta_test,
        }

        save_to = save_to or self.data_path
        file = os.path.join(save_to, "sh30m.pv.ds.pkl")
        if os.path.exists(file):
            now = arrow.now().format("YYMMDD")
            file = os.path.join(save_to, "sh30m.pv.ds.", now, ".pkl")
        with open(file, "wb") as f:
            pickle.dump(data, f)

    def train(self, data=None, save_to=None):
        """训练模型"""
        if data is None:
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
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
            n_iter=100,
            cv=3,
            verbose=1,
            n_jobs=psutil.cpu_count() - 1,
            return_train_score=True,
            refit=True,
        )

        search.fit(X_train, y_train)
        model = search.best_estimator_

        preds = model.predict(X_test)
        report = classification_report(y_test, preds)

        print(report)

        save_to = save_to or self.model_path
        file = os.path.join(save_to, "sh30m.pv.xgb.pkl")
        if os.path.exists(file):
            now = arrow.now().format("YYMMDD")
            file = os.path.join(save_to, "sh30m.pv.xgb.", now, ".pkl")

        with open(file, "wb") as f:
            pickle.dump(model, f)

        desc_file = os.path.join(save_to, "sh30m.pv.desc")
        if os.path.exists(desc_file):
            now = arrow.now().format("YYMMDD")
            desc_file = os.path.join(save_to, "sh30m.pv.", now, ".desc")
        with open(desc_file, "w") as f:
            f.write(report)

        return model, report

    def predict(self, bars: np.ndarray):
        features, _ = reversal_features(self.code, bars, FrameType.MIN30)
        label = self.model.predict(np.array([features]))[0]
        return (label, {0: "顶部反转", 1: "底部反转", 2: "趋势延续"}.get(label))

    async def watch(self):
        """监控模型"""
        bars = await get_bars(self.code, 100, "30m")
        return self.predict(bars)


if __name__ == "__main__":
    import asyncio

    import cfg4py
    import omicron

    from alpha.config import get_config_dir

    cfg4py.init(get_config_dir())

    async def main():
        await omicron.init()
        pv = IndexShPeakValleys(version="v1")
        os.makedirs(f"/tmp/alpha/", exist_ok=True)
        save_to = f"/tmp/alpha/"
        await pv.make_dataset(save_to)

        ds = f"/tmp/alpha/sh30m.pv.ds.pkl"
        data = pickle.load(open(ds, "rb"))
        pv.train(data, save_to=f"/tmp/alpha/")

    asyncio.run(main())
