import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# data stack for ploting.
train_acc = []
train_err = []
valid_acc = []
valid_err = []


# TODO: LOCAL MINIMA PROBLEM, DYING RELU PROBLEM. so change the seed value.


# plotting func.
def plotting(train_acc, train_err, valdiate_acc, validate_err):
    train_acc = np.array(train_acc)
    train_err = np.array(train_err)
    valdiate_acc = np.array(valdiate_acc)
    validate_err = np.array(validate_err)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.set_xlabel("accuracy")
    ax1.set_ylim([0, 1])
    ax1.plot(train_acc, "r", label='train')
    ax1.plot(valdiate_acc, "g", label='test')
    ax1.legend(loc='lower right')

    ax2.set_xlabel("loss")
    ax2.set_ylim([0, 0.03])
    ax2.plot(train_err, "r", label='train')
    ax2.plot(validate_err, "g", label='test')
    ax2.legend(loc='upper right')

    plt.show()

    # Network model class.


class NeuralNetwork:
    # config data.
    TOTAL_EPOCH = 500
    BATCH_SIZE = 60000
    LEARNING_RATE = 0.000001
    # LEARNING_RATE = 0.1
    # LEARNING_RATE = 0.1/60000
    SEED = 42
    TRAIN_DATASET_SIZE = 60000
    TEST_DATASET_SIZE = 10000

    def __init__(self, learning_rate):
        # for batch count.
        self._i = 0

        # weight initialize
        # because of dying relu problem, divide to large number.
        self.w1 = np.random.randn(784, 100) / 10
        self.w2 = np.random.randn(100, 50) / 10
        self.w3 = np.random.randn(50, 10) / 10
        self.learningRate = learning_rate

        # load data using scikit-learn.
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(one_hot=True)
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255

    def relu(self, x):
        back_relu = np.zeros(x.shape)
        back_relu[np.where(x > 0)] = 1
        x[np.where(x <= 0)] = 0
        return x, back_relu

    def softmax(self, x):
        if x.ndim == 1:
            x = x.reshape([1, x.size])
        modifiedX = x - np.max(x, 1).reshape([x.shape[0], 1])
        sigmoid = np.exp(modifiedX)
        return sigmoid / np.sum(sigmoid, axis=1).reshape([sigmoid.shape[0], 1])

    def feedForward_sigmoid(self, x):
        y1 = np.dot(x, self.w1)
        w1 = self.w1
        activated_y1, back_relu_w1 = self.relu(y1)

        y2 = np.dot(activated_y1, self.w2)
        w2 = self.w2
        activated_y2, back_relu_w2 = self.relu(y2)

        y3 = np.dot(activated_y2, self.w3)
        w2 = self.w3
        softmax_result = self.softmax(y3)

        return activated_y1, activated_y2, softmax_result, back_relu_w1, back_relu_w2

    def backpropagation_sigmoid(self, x, labelY, predictY1, predictY2, predictY3, back_relu_w1, back_relu_w2):
        e = (predictY3 - labelY)
        d_w3 = np.matmul(predictY2.T, e)
        d_w2 = np.matmul(predictY1.T, np.matmul(e, self.w3.T) * back_relu_w2)
        d_w1 = np.matmul(x.T, np.matmul(np.matmul(e, self.w3.T) * back_relu_w2, self.w2.T) * back_relu_w1)

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


# fix the random value.
np.random.seed(NeuralNetwork.SEED)

# define network model.
network_model = NeuralNetwork(learning_rate=NeuralNetwork.LEARNING_RATE)

# using mini-batch
for i in range(NeuralNetwork.TOTAL_EPOCH):
    print("============== EPOCH {} START ==============".format(i + 1))
    for j in range(NeuralNetwork.TRAIN_DATASET_SIZE // NeuralNetwork.BATCH_SIZE):
        print("-------------- batch {} training...".format(j))
        batch_x, batch_y = network_model.next_batch(NeuralNetwork.BATCH_SIZE)

        # TODO: feed-forward term.
        y1, y2, y3, back_relu_w1, back_relu_w2 = network_model.feedForward_sigmoid(batch_x)

        # TODO: back-propagation term.
        dW1, dW2, dW3 = network_model.backpropagation_sigmoid(batch_x, batch_y, y1, y2, y3, back_relu_w1, back_relu_w2)
        network_model.update_weight(dW1, dW2, dW3)

    print("============== EPOCH {} END ================".format(i + 1))

    # shake data when epoch ended.
    # network_model.shake_data()

    # add one epoch data.
    _, _, y_train3, _, _ = network_model.feedForward_sigmoid(network_model.X_train)
    match_prediction_train = np.equal(np.argmax(y_train3, axis=1),
                                      np.argmax(network_model.y_train, axis=1))
    accuracy_train = np.mean(match_prediction_train)
    match_loss_train = (1 / 2) * ((y_train3 - network_model.y_train) ** 2)

    train_acc.append(accuracy_train)
    train_err.append(np.mean(match_loss_train))

    # calculate test dataset.
    _, _, y_test3, _, _ = network_model.feedForward_sigmoid(network_model.X_test)
    match_prediction_test = np.equal(np.argmax(y_test3, axis=1), np.argmax(network_model.y_test, axis=1))
    accuracy_test = np.mean(match_prediction_test)
    match_loss_test = (1 / 2) * ((y_test3 - network_model.y_test) ** 2)

    valid_acc.append(accuracy_test)
    valid_err.append(np.mean(match_loss_test))
    print("train accuracy : {:.4}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(accuracy_train,
                                                                                             np.mean(match_loss_train),
                                                                                             accuracy_test,
                                                                                             np.mean(match_loss_test)))

    # draw graph.
    plotting(train_acc, train_err, valid_acc, valid_err)
