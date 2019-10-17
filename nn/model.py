import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml


class NeuralNetwork:
    OPTIMIZER_ADAM = "adam"
    OPTIMIZER_ADAGRAD = "adagrad"
    ACTIVATE_RELU = "relu"
    ACTIVATE_SIGMOID = "sigmoid"

    def __init__(self, configure, h1=100, h2=50):
        # for batch count.
        self._i = 0

        # weight initialize
        self.w1 = np.random.randn(784, h1)
        self.w2 = np.random.randn(h1, h2)
        self.w3 = np.random.randn(h2, 10)

        # config data.
        self.OTAL_EPOCH = 120
        self.BATCH_SIZE = 1000
        self.LEARNING_RATE = 0.001
        self.SEED = 42
        self.TRAIN_DATASET_SIZE = 60000
        self.TEST_DATASET_SIZE = 10000
        self.OPTIMIZER_TYPE = "adam"
        self.ACTIVATE_TYPE = "relu"
        # self.OTAL_EPOCH = 120
        # self.BATCH_SIZE = 1000
        # self.LEARNING_RATE = 0.001
        # self.SEED = 42
        # self.TRAIN_DATASET_SIZE = 60000
        # self.TEST_DATASET_SIZE = 10000
        # self.OPTIMIZER_TYPE = "adam"
        # self.ACTIVATE_TYPE = "relu"

        # load data using scikit-learn.
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(one_hot=True)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def back_sigmoid(self, x):
        return x * (1. - x)

    def softmax(self, x):
        if x.ndim == 1:
            x = x.reshape([1, x.size])
        modifiedX = x - np.max(x, 1).reshape([x.shape[0], 1])
        sigmoid = np.exp(modifiedX)
        return sigmoid / np.sum(sigmoid, axis=1).reshape([sigmoid.shape[0], 1])

    def feedForward_sigmoid(self, x):
        y1 = np.dot(x, self.w1)
        activated_y1 = self.sigmoid(y1)

        y2 = np.dot(activated_y1, self.w2)
        activated_y2 = self.sigmoid(y2)

        y3 = np.dot(activated_y2, self.w3)
        softmax_result = self.softmax(y3)

        return activated_y1, activated_y2, softmax_result

    def backpropagation_sigmoid(self, x, labelY, predictY1, predictY2, predictY3):
        e = (predictY3 - labelY) / NeuralNetwork.BATCH_SIZE
        d_w3 = predictY2.T.dot(e)
        d_w2 = predictY1.T.dot(np.matmul(e, self.w3.T) * self.back_sigmoid(predictY2))
        d_w1 = x.T.dot(np.matmul(np.matmul(e, self.w3.T) * self.back_sigmoid(predictY2),
                                 self.w2.T) * self.back_sigmoid(predictY1))

        return d_w1, d_w2, d_w3

    def update_weight(self, d_w1, d_w2, d_w3):
        self.w1 -= self.learningRate * d_w1
        self.w2 -= self.learningRate * d_w2
        self.w3 -= self.learningRate * d_w3

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

        return X[:NeuralNetwork.TRAIN_DATASET_SIZE], X[NeuralNetwork.TRAIN_DATASET_SIZE:], y[
                                                                                           :NeuralNetwork.TRAIN_DATASET_SIZE], y[
                                                                                                                               NeuralNetwork.TRAIN_DATASET_SIZE:]

    # print next batch.
    def next_batch(self, batch_size):
        x, y = self.X_train[self._i:self._i + batch_size], self.y_train[self._i: self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.X_train)
        return x, y

    def shake_data(self):
        rnd_index = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[rnd_index]
        self.y_train = self.y_train[rnd_index]
