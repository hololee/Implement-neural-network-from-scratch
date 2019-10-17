import numpy as np

OPTIMIZER_GD = "gradient_descent"
OPTIMIZER_ADAM = "adam"
OPTIMIZER_ADAGRAD = "adagrad"
ACTIVATE_RELU = "relu"
ACTIVATE_SIGMOID = "sigmoid"


class NeuralNetwork:

    def __init__(self, configure, h1=100, h2=50):
        # weight initialize
        self.w1 = np.random.randn(784, h1)
        self.w2 = np.random.randn(h1, h2)
        self.w3 = np.random.randn(h2, 10)

        # config data.
        self.TOTAL_EPOCH = configure['total_epoch']
        self.BATCH_SIZE = configure['batch_size']
        self.LEARNING_RATE = configure['learning_rate']
        self.SEED = configure['random_seed']
        self.OPTIMIZER = configure['optimizer']
        self.ACTIVATION = configure['activation']

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def back_sigmoid(self, x):
        return x * (1. - x)

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

    def feedForward(self, x):
        y1 = np.dot(x, self.w1)
        if self.ACTIVATION == ACTIVATE_SIGMOID:
            activated_y1 = self.sigmoid(y1)
            back_relu_w1 = None
        elif self.ACTIVATION == ACTIVATE_RELU:
            activated_y1, back_relu_w1 = self.relu(y1)
        else:
            activated_y1 = self.sigmoid(y1)
            back_relu_w1 = None

        y2 = np.dot(activated_y1, self.w2)
        if self.ACTIVATION == ACTIVATE_SIGMOID:
            activated_y2 = self.sigmoid(y2)
            back_relu_w2 = None
        elif self.ACTIVATION == ACTIVATE_RELU:
            activated_y2, back_relu_w2 = self.relu(y2)
        else:
            activated_y2 = self.sigmoid(y2)
            back_relu_w2 = None

        y3 = np.dot(activated_y2, self.w3)
        softmax_result = self.softmax(y3)

        return activated_y1, activated_y2, softmax_result, back_relu_w1, back_relu_w2

    def backpropagation(self, x, labelY, out1, out2, out3, back_relu_w1, back_relu_w2):
        d_e = (out3 - labelY) / self.BATCH_SIZE

        # calculate d_w3
        d_w3 = out2.T.dot(d_e)

        # calculate d_w2
        if self.ACTIVATION == ACTIVATE_SIGMOID:
            d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2))
        elif self.ACTIVATION == ACTIVATE_RELU:
            d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * back_relu_w2)
        else:
            d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2))

        # calculate d_w1
        if self.ACTIVATION == ACTIVATE_SIGMOID:
            d_w1 = x.T.dot(
                np.matmul(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2), self.w2.T) * self.back_sigmoid(out1))
        elif self.ACTIVATION == ACTIVATE_RELU:
            d_w1 = x.T.dot(np.matmul(np.matmul(d_e, self.w3.T) * back_relu_w2, self.w2.T) * back_relu_w1)
        else:
            d_w1 = x.T.dot(
                np.matmul(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2), self.w2.T) * self.back_sigmoid(out1))

        # return changed value.
        return d_w1, d_w2, d_w3

    def update_weight(self, d_w1, d_w2, d_w3):
        self.w1 -= self.LEARNING_RATE * d_w1
        self.w2 -= self.LEARNING_RATE * d_w2
        self.w3 -= self.LEARNING_RATE * d_w3

    def train(self, input, output):
        # TODO: feed-forward term.
        y1, y2, y3, back_relu_w1, back_relu_w2 = self.feedForward(input)
        # TODO: back-propagation term.
        dW1, dW2, dW3 = self.backpropagation(input, output, y1, y2, y3, back_relu_w1, back_relu_w2)
        self.update_weight(dW1, dW2, dW3)

    def predict(self, input):
        _, _, output, _, _ = self.feedForward(input)
        return output

    def getAccuracyAndLoss(self, output_of_model, output):
        accuracy = np.mean(np.equal(np.argmax(output_of_model, axis=1),
                                    np.argmax(output, axis=1)))
        # l2 loss
        loss = np.mean((1 / 2) * ((output_of_model - output) ** 2))

        return accuracy, loss
