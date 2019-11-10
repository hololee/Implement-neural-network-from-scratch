from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
import numpy as np


class DataManager:
    def __init__(self):
        # for batch count.
        self._i = 0

        # set divide size.
        self.train_dataset_size = 60000
        self.test_dataset_size = 10000

        # load data using scikit-learn.
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(one_hot=True)

        # change range 0 to 1.
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255

        self.X_train[np.where(self.X_train == 0)] = 0.01
        self.X_train[np.where(self.X_train == 1)] = 0.999
        self.X_test[np.where(self.X_test == 0)] = 0.01
        self.X_test[np.where(self.X_test == 1)] = 0.999
        self.y_train[np.where(self.y_train == 0)] = 0.01
        self.y_train[np.where(self.y_train == 1)] = 0.999
        self.y_test[np.where(self.y_test == 0)] = 0.01
        self.y_test[np.where(self.y_test == 1)] = 0.999
        pass


    def load_data(self, one_hot=True):
        print("loading data...")

        # only use scikit-learn when load MNIST data for convenience
        mnist = fetch_openml('mnist_784')
        X, y = mnist["data"], mnist["target"]

        if one_hot:
            one_hot = OneHotEncoder()
            y = one_hot.fit_transform(y.reshape(-1, 1))
            y = y.toarray()
            print("y are one-hot encoded..")

        return X[:self.train_dataset_size], X[self.train_dataset_size:], y[:self.train_dataset_size], y[
                                                                                                      self.train_dataset_size:]

    # print next batch.
    def next_batch(self, batch_size):
        x, y = self.X_train[self._i:self._i + batch_size], self.y_train[self._i: self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.X_train)
        return x, y

    def shake_data(self):
        rnd_index = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[rnd_index]
        self.y_train = self.y_train[rnd_index]
