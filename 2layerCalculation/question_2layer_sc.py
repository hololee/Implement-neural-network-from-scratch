import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml


# Network nn class.
class NeuralNetwork:
    # config data.
    TOTAL_EPOCH = 15
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.001
    SEED = 42
    TRAIN_DATASET_SIZE = 60000
    TEST_DATASET_SIZE = 10000

    def __init__(self, learning_rate):
        # for batch count.
        self._i = 0

        # weight initialize
        self.W1 = np.random.randn(784, 100)
        self.W2 = np.random.randn(100, 10)
        self.learningRate = learning_rate

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

    def getCrossEntropy(self, predictY, labelY):
        return np.mean(-np.sum(labelY * np.log(self.softmax(predictY)), axis=1))

    def feedForward(self, x):
        y1 = np.dot(x, self.W1)
        sigmoidY1 = self.sigmoid(y1)

        y2 = np.dot(sigmoidY1, self.W2)
        softmaxY2 = self.softmax(y2)

        return sigmoidY1, softmaxY2

    def backpropagation(self, x, labelY, predictY1, predictY2):
        dW2 = predictY1.T.dot((predictY2 - labelY) / NeuralNetwork.BATCH_SIZE)
        dW1 = x.T.dot(np.matmul(predictY2 - labelY, self.W2.T) * self.back_sigmoid(predictY1))

        return dW1, dW2

    def update_weight(self, dW1, dW2):
        self.W1 -= self.learningRate * dW1
        self.W2 -= self.learningRate * dW2

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


# fix the random value.
np.random.seed(NeuralNetwork.SEED)

# define network nn.
network_model = NeuralNetwork(learning_rate=NeuralNetwork.LEARNING_RATE)

# using mini-batch
for i in range(NeuralNetwork.TOTAL_EPOCH):
    print("============== EPOCH {} START ==============".format(i + 1))
    for j in range(NeuralNetwork.TRAIN_DATASET_SIZE // NeuralNetwork.BATCH_SIZE):
        print("-------------- batch {} training...".format(j))
        batch_x, batch_y = network_model.next_batch(NeuralNetwork.BATCH_SIZE)

        # TODO: feed-forward term.
        y1, y2 = network_model.feedForward(batch_x)

        # TODO: back-propagation term.
        dW1, dW2 = network_model.backpropagation(batch_x, batch_y, y1, y2)
        network_model.update_weight(dW1, dW2)
    print("============== EPOCH {} END ================".format(i + 1))

    # shake data.
    network_model.shake_data()

    y1, y2 = network_model.feedForward(network_model.X_test)
    correct_prediction = np.equal(np.argmax(y2, axis=1), np.argmax(network_model.y_test, axis=1))
    accuracy = np.mean(correct_prediction)
    print(accuracy)
