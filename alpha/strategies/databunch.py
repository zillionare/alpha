from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.utils import shuffle
import pickle

class DataBunch(Bunch):
    """
    A data bunch is a collection of data sources, which are used to
    bunch up data for training.
    """
    def __init__(self, name=None, data=None, target=None):
        self.data = None
        self.target = None
        self.name = None

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

    def __getitem__(self, key):
        return self.data[key], self.target[key]

    def train_test_split(self, test_size=0.2, random_state=78):
        """
        Splits data into train and test sets, and returns as numpy arrays.
        """
        X = self.data
        y = self.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

def load_data(dataset_path:str)->DataBunch:
    """
    Loads data from a given dataset path.
    """
    with open(dataset_path, 'rb') as f:
        bunch = pickle.load(f)

    return bunch
