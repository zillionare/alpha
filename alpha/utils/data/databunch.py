import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, shuffle

from alpha.core.features import moving_average


class DataBunch(Bunch):
    """
    A data bunch is a collection of data sources, which are used to
    bunch up data for training.
    """

    def __init__(
        self, X=None, y=None, raw=None, name=None, desc: str = None, version: str = None
    ):
        self.data = np.array(X)
        self.target = np.array(y)
        self.raw = raw

        self.name = name
        self.desc = desc
        self.version = version

        self.path = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def shuffle(self, random_state=None):
        """shuffle data/target and keeps other keys intact"""

        X = self.data
        y = self.target

        X, y = shuffle(X, y, random_state=random_state)

        self.data = X
        self.target = y

    def __repr__(self):
        return f"{self.name} ({self.data.shape})"

    def __str__(self):
        return f"{self.name} ({self.data.shape})"

    def __len__(self):
        return len(self.data)

    def train_test_split(self, test_size=0.2, random_state=78, stratified=True):
        """
        Splits data into train and test sets, and returns as numpy arrays.
        """
        X = self.data
        y = self.target

        if stratified:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __get_state__(self):
        """don't pickle path

        Returns:
            [type]: [description]
        """
        state = self.__dict__
        del state["path"]
        return state

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @property
    def X(self):
        return self.data

    @property
    def y(self):
        return self.target

    def plot(self, i, wins=[5, 10, 20, 60], convas_win=30, ax=None):
        code, xbars, ybars = self.raw[i]
        import matplotlib.pyplot as plt

        xclose = xbars["close"]
        yclose = ybars["close"]

        close = np.concatenate((xclose, yclose))
        for win in wins:
            ma = moving_average(close, win)[-convas_win:]
            plt.plot(ma, label=f"{win}d")

        yend = convas_win
        ystart = yend - len(ybars)

        xclose_bars = convas_win - len(ybars)
        xend = ystart
        xstart = xend - xclose_bars

        plt.plot(np.arange(ystart, yend), yclose, "r.", label="close")
        plt.plot(np.arange(xstart, xend), xclose[-xclose_bars:], "b.")

        if ax:
            transform = ax.transAxes
        else:
            transform = plt.gca().transAxes
        plt.text(0, 0.95, f"{code} {xbars['frame'][-1]}", transform=transform)
