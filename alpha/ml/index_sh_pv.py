"""大盘30分钟顶底预测
"""
import pickle
import psutil

from sklearn.metrics import classification_report
from alpha.core.features import peaks_and_valleys, reversal_features
from alpha.notebook import get_bars
import os
import random
import arrow
import numpy as np
from omicron.core.types import FrameType

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


class IndexShPeakValleys:
    def __init__(self, inference_mode=False):
        self.code = "000001.XSHG"
        self.data_path = os.path.expanduser(
            "~/zillionare/alpha/data/"
        )
        self.model_path = os.path.expanduser(
            "~/zillionare/alpha/models/"
        )

        if inference_mode:
            file = os.path.join(self.model_path, "sh30m.pv.xgb.pkl")
            with open(file, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = None

    async def make_dataset(self, cpr=1, save_to=None):
        """制作顶底的标注数据"""
        bars = await get_bars(self.code, 12000, "30m")
        peaks, valleys = peaks_and_valleys(bars["close"])

        features = []
        for i in peaks:
            if i < 100 or i == len(bars) - 1:
                continue

            bars_ = bars[i - 99 : i + 1]
            feature, _ = reversal_features(self.code, bars_, FrameType.MIN30)
            features.append([*feature, 0, i])

        for i in valleys:
            if i < 100 or i == len(bars) - 1:
                continue

            bars_ = bars[i - 99 : i + 1]
            feature, _ = reversal_features(self.code, bars_, FrameType.MIN30)
            features.append([*feature, 1, i])

        pv = set(peaks)
        pv.update(set(valleys))
        count_of_middle = int(len(pv) * cpr)
        middle = list(set(np.arange(100, len(bars) - 1)).difference(pv))
        for i in random.sample(middle, count_of_middle):
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
        return label, {
            0: "顶部反转",
            1: "底部反转",
            2: "趋势延续",
        }.get(label)

    async def watch(self):
        """监控模型"""
        bars = await get_bars(self.code, 100, "30m")
        return self.predict(bars)

if __name__ == "__main__":
    import asyncio
    import omicron
    from alpha.config import get_config_dir
    import cfg4py

    cfg4py.init(get_config_dir())
    async def main():
        await omicron.init()
        pv = IndexShPeakValleys()
        for cpr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            os.makedirs(f"/tmp/alpha/{cpr}", exist_ok=True)
            save_to = f"/tmp/alpha/{cpr}/pv.ds.pkl"
            await pv.make_dataset(cpr, save_to)

            data = pickle.load(open(save_to, "rb"))
            print(f"CPR is {cpr}")
            pv.train(data, save_to=f"/tmp/alpha/{cpr}/pv.model.pkl")

    asyncio.run(main())




