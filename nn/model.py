import numpy as np

OPTIMIZER_GD = "gradient_descent"
OPTIMIZER_GD_MOMENTUM = "gradient_descent_using_momentum"
OPTIMIZER_ADAM = "adam"
OPTIMIZER_ADAGRAD = "adagrad"
ACTIVATE_RELU = "relu"
ACTIVATE_SIGMOID = "sigmoid"
LOSS_MSE = "mse"
LOSS_CROSSENTROPY = "cross_entropy"


class NeuralNetwork:

    def __init__(self, configure, h1=100, h2=50):
        # weight initialize : Xavier initialization
        self.w1 = np.random.randn(784, h1) / np.sqrt(784)
        self.w2 = np.random.randn(h1, h2) / np.sqrt(h1)
        self.w3 = np.random.randn(h2, 10) / np.sqrt(h2)

        self.b1 = np.random.randn(1, h1) / np.sqrt(784)
        self.b2 = np.random.randn(1, h2) / np.sqrt(h1)
        self.b3 = np.random.randn(1, 10) / np.sqrt(h2)

        # set configure.
        self.configure = configure

        # config data.
        self.TOTAL_EPOCH = configure['total_epoch']
        self.BATCH_SIZE = configure['batch_size']
        self.LEARNING_RATE = configure['learning_rate']
        self.SEED = configure['random_seed']
        self.OPTIMIZER = configure['optimizer']
        self.ACTIVATION = configure['activation']
        self.LOSS = configure['loss']

        if self.OPTIMIZER == OPTIMIZER_GD_MOMENTUM:
            # momenum
            self.MOMENTUM = configure['momentum']
            self.prev_dW1 = np.zeros(self.w1.shape)
            self.prev_dW2 = np.zeros(self.w2.shape)
            self.prev_dW3 = np.zeros(self.w3.shape)
            self.prev_db1 = np.zeros(self.b1.shape)
            self.prev_db2 = np.zeros(self.b2.shape)
            self.prev_db3 = np.zeros(self.b3.shape)

        if self.OPTIMIZER == OPTIMIZER_ADAGRAD:
            self.eps = configure['epsilon']
            self.gt_w1 = np.zeros(self.w1.shape)
            self.gt_w2 = np.zeros(self.w2.shape)
            self.gt_w3 = np.zeros(self.w3.shape)
            self.gt_b1 = np.zeros(self.b1.shape)
            self.gt_b2 = np.zeros(self.b2.shape)
            self.gt_b3 = np.zeros(self.b3.shape)

        if self.OPTIMIZER == OPTIMIZER_ADAM:
            self.beta1 = configure['beta1']
            self.beta2 = configure['beta2']
            self.eps = configure['epsilon']

            # for calculate beta.
            self.counts = 1

            self.mt_w1 = np.zeros(self.w1.shape)
            self.vt_w1 = np.zeros(self.w1.shape)
            self.mt_b1 = np.zeros(self.b1.shape)
            self.vt_b1 = np.zeros(self.b1.shape)

            self.mt_w2 = np.zeros(self.w2.shape)
            self.vt_w2 = np.zeros(self.w2.shape)
            self.mt_b2 = np.zeros(self.b2.shape)
            self.vt_b2 = np.zeros(self.b2.shape)

            self.mt_w3 = np.zeros(self.w3.shape)
            self.vt_w3 = np.zeros(self.w3.shape)
            self.mt_b3 = np.zeros(self.b3.shape)
            self.vt_b3 = np.zeros(self.b3.shape)

    def sigmoid(self, x):
        array = np.copy(x)
        return 1.0 / (1.0 + np.exp(-array))

    def back_sigmoid(self, x):
        array = np.copy(x)
        return array * (1 - array)

    def relu(self, x):
        array = np.copy(x)
        array[array < 0] = 0
        return array

    def back_relu(self, x):
        array = np.copy(x)
        array[array >= 0] = 1
        array[array < 0] = 0
        return array

    def softmax(self, x):
        array = np.copy(x)
        if array.ndim == 1:
            array = array.reshape([1, array.size])
        exps = np.exp(array)
        return exps / np.sum(exps, axis=1).reshape([exps.shape[0], 1])

    def feedForward(self, x):

        if self.LOSS == LOSS_CROSSENTROPY:
            y1 = np.dot(x, self.w1) + self.b1
            if self.ACTIVATION == ACTIVATE_SIGMOID:
                activated_y1 = self.sigmoid(y1)

                y2 = np.dot(activated_y1, self.w2) + self.b2
                activated_y2 = self.sigmoid(y2)

                y3 = np.dot(activated_y2, self.w3) + self.b3
                result = self.softmax(y3)

                return activated_y1, activated_y2, result

            elif self.ACTIVATION == ACTIVATE_RELU:
                activated_y1 = self.relu(y1)
                y2 = np.dot(activated_y1, self.w2) + self.b2
                activated_y2 = self.relu(y2)
                y3 = np.dot(activated_y2, self.w3) + self.b3
                result = self.softmax(y3)

                return activated_y1, activated_y2, result

        elif self.LOSS == LOSS_MSE:
            y1 = np.dot(x, self.w1) + self.b1
            if self.ACTIVATION == ACTIVATE_SIGMOID:
                activated_y1 = self.sigmoid(y1)

                y2 = np.dot(activated_y1, self.w2) + self.b2
                activated_y2 = self.sigmoid(y2)

                y3 = np.dot(activated_y2, self.w3) + self.b3
                activated_y3 = self.sigmoid(y3)
                result = activated_y3

                return activated_y1, activated_y2, result

            elif self.ACTIVATION == ACTIVATE_RELU:
                activated_y1 = self.relu(y1)
                y2 = np.dot(activated_y1, self.w2) + self.b2
                activated_y2 = self.relu(y2)
                y3 = np.dot(activated_y2, self.w3) + self.b3
                activated_y3 = self.relu(y3)
                result = activated_y3

                return activated_y1, activated_y2, result

    def backpropagation(self, x, labelY, out1, out2, out3):

        if self.LOSS == LOSS_CROSSENTROPY:
            d_e = (out3 - labelY)
            # calculate d_w3
            d_w3 = out2.T.dot(d_e)
            d_b3 = np.ones(shape=[1, self.BATCH_SIZE]).dot(d_e)

            if self.ACTIVATION == ACTIVATE_SIGMOID:
                # calculate d_w2
                d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2))
                d_b2 = np.ones(shape=[1, self.BATCH_SIZE]).dot(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2))
                # calculate d_w1
                d_w1 = x.T.dot(
                    np.matmul(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2), self.w2.T) * self.back_sigmoid(out1))
                d_b1 = np.ones(shape=[1, self.BATCH_SIZE]).dot(
                    np.matmul(np.matmul(d_e, self.w3.T) * self.back_sigmoid(out2), self.w2.T) * self.back_sigmoid(
                        out1))
                return d_w1, d_w2, d_w3, d_b1, d_b2, d_b3

            elif self.ACTIVATION == ACTIVATE_RELU:
                d_w2 = out1.T.dot(np.matmul(d_e, self.w3.T) * self.back_relu(out2))
                d_b2 = np.ones(shape=[1, self.BATCH_SIZE]).dot(
                    np.matmul(d_e, self.w3.T) * self.back_relu(out2))
                d_w1 = x.T.dot(np.matmul(np.matmul(d_e, self.w3.T) * self.back_relu(out2),
                                         self.w2.T) * self.back_relu(out1))
                d_b1 = np.ones(shape=[1, self.BATCH_SIZE]).dot(
                    np.matmul(np.matmul(d_e, self.w3.T) * self.back_relu(out2),
                              self.w2.T) * self.back_relu(out1))
                return d_w1, d_w2, d_w3, d_b1, d_b2, d_b3

        elif self.LOSS == LOSS_MSE:
            e = (out3 - labelY)
            if self.ACTIVATION == ACTIVATE_SIGMOID:
                # calculate d_w3
                d_w3 = out2.T.dot(e * self.back_sigmoid(out3))
                d_b3 = np.ones(shape=[1, self.BATCH_SIZE]).dot(e * self.back_sigmoid(out3))
                # calculate d_w2
                d_w2 = out1.T.dot(np.dot(e * self.back_sigmoid(out3), self.w3.T) * self.back_sigmoid(out2))
                d_b2 = np.ones(shape=[1, self.BATCH_SIZE]).dot(
                    np.dot(e * self.back_sigmoid(out3), self.w3.T) * self.back_sigmoid(out2))
                # calculate d_w1
                d_w1 = x.T.dot(np.dot(np.dot(e * self.back_sigmoid(out3), self.w3.T) * self.back_sigmoid(out2),
                                      self.w2.T) * self.back_sigmoid(out1))
                d_b1 = np.ones(shape=[1, self.BATCH_SIZE]).dot(
                    np.dot(np.dot(e * self.back_sigmoid(out3), self.w3.T) * self.back_sigmoid(out2),
                           self.w2.T) * self.back_sigmoid(out1))

                return d_w1, d_w2, d_w3, d_b1, d_b2, d_b3

            elif self.ACTIVATION == ACTIVATE_RELU:
                # calculate d_w3
                d_w3 = out2.T.dot(e * self.back_relu(out3))
                d_b3 = np.ones(shape=[1, self.BATCH_SIZE]).dot(e * self.back_relu(out3))
                # calculate d_w2
                d_w2 = out1.T.dot(
                    np.dot(e * self.back_relu(out3), self.w3.T) * self.back_relu(out2))
                d_b2 = np.ones(shape=[1, self.BATCH_SIZE]).dot(
                    np.dot(e * self.back_relu(out3), self.w3.T) * self.back_relu(out2))
                # calculate d_w1
                d_w1 = x.T.dot(np.dot(
                    np.dot(e * self.back_relu(out3), self.w3.T) * self.back_relu(out2),
                    self.w2.T) * self.back_relu(out1))
                d_b1 = np.ones(shape=[1, self.BATCH_SIZE]).dot(
                    np.dot(np.dot(e * self.back_relu(out3), self.w3.T) * self.back_relu(
                        out2), self.w2.T) * self.back_relu(out1))

                return d_w1, d_w2, d_w3, d_b1, d_b2, d_b3

    def update_weight(self, d_w1, d_w2, d_w3, d_b1, d_b2, d_b3):
        if self.OPTIMIZER == OPTIMIZER_GD:
            self.w1 -= self.LEARNING_RATE * d_w1
            self.w2 -= self.LEARNING_RATE * d_w2
            self.w3 -= self.LEARNING_RATE * d_w3
            self.b1 -= self.LEARNING_RATE * d_b1
            self.b2 -= self.LEARNING_RATE * d_b2
            self.b3 -= self.LEARNING_RATE * d_b3

        elif self.OPTIMIZER == OPTIMIZER_GD_MOMENTUM:
            self.prev_dW1 = (self.MOMENTUM * self.prev_dW1) + (self.LEARNING_RATE * d_w1)
            self.prev_dW2 = (self.MOMENTUM * self.prev_dW2) + (self.LEARNING_RATE * d_w2)
            self.prev_dW3 = (self.MOMENTUM * self.prev_dW3) + (self.LEARNING_RATE * d_w3)
            self.prev_db1 = (self.MOMENTUM * self.prev_db1) + (self.LEARNING_RATE * d_b1)
            self.prev_db2 = (self.MOMENTUM * self.prev_db2) + (self.LEARNING_RATE * d_b2)
            self.prev_db3 = (self.MOMENTUM * self.prev_db3) + (self.LEARNING_RATE * d_b3)

            self.w1 -= self.prev_dW1
            self.w2 -= self.prev_dW2
            self.w3 -= self.prev_dW3
            self.b1 -= self.prev_db1
            self.b2 -= self.prev_db2
            self.b3 -= self.prev_db3

        elif self.OPTIMIZER == OPTIMIZER_ADAGRAD:
            # update the gt.
            self.gt_w1 += np.square(d_w1 ** 2)
            self.gt_w2 += np.square(d_w2 ** 2)
            self.gt_w3 += np.square(d_w3 ** 2)
            self.gt_b1 += np.square(d_b1 ** 2)
            self.gt_b2 += np.square(d_b2 ** 2)
            self.gt_b3 += np.square(d_b3 ** 2)

            # change the learning rate for each weight.
            self.w1 -= (self.LEARNING_RATE / np.sqrt(self.gt_w1 + self.eps)) * d_w1
            self.w2 -= (self.LEARNING_RATE / np.sqrt(self.gt_w2 + self.eps)) * d_w2
            self.w3 -= (self.LEARNING_RATE / np.sqrt(self.gt_w3 + self.eps)) * d_w3
            # change the learning rate for each bias.
            self.b1 -= (self.LEARNING_RATE / np.sqrt(self.gt_b1 + self.eps)) * d_b1
            self.b2 -= (self.LEARNING_RATE / np.sqrt(self.gt_b2 + self.eps)) * d_b2
            self.b3 -= (self.LEARNING_RATE / np.sqrt(self.gt_b3 + self.eps)) * d_b3

        elif self.OPTIMIZER == OPTIMIZER_ADAM:

            self.mt_w1 = (self.beta1 * self.mt_w1) + ((1 - self.beta1) * d_w1)
            self.vt_w1 = (self.beta2 * self.vt_w1) + ((1 - self.beta2) * (d_w1 ** 2))
            self.mt_b1 = (self.beta1 * self.mt_b1) + ((1 - self.beta1) * d_b1)
            self.vt_b1 = (self.beta2 * self.vt_b1) + ((1 - self.beta2) * (d_b1 ** 2))

            self.mt_w1 = self.mt_w1 / (1 - self.beta1)
            self.vt_w1 = self.vt_w1 / (1 - self.beta2)
            self.mt_b1 = self.mt_b1 / (1 - self.beta1)
            self.vt_b1 = self.vt_b1 / (1 - self.beta2)

            self.mt_w2 = (self.beta1 * self.mt_w2) + ((1 - self.beta1) * d_w2)
            self.vt_w2 = (self.beta2 * self.vt_w2) + ((1 - self.beta2) * (d_w2 ** 2))
            self.mt_b2 = (self.beta1 * self.mt_b2) + ((1 - self.beta1) * d_b2)
            self.vt_b2 = (self.beta2 * self.vt_b2) + ((1 - self.beta2) * (d_b2 ** 2))

            self.mt_w2 = self.mt_w2 / (1 - self.beta1)
            self.vt_w2 = self.vt_w2 / (1 - self.beta2)
            self.mt_b2 = self.mt_b2 / (1 - self.beta1)
            self.vt_b2 = self.vt_b2 / (1 - self.beta2)

            self.mt_w3 = (self.beta1 * self.mt_w3) + ((1 - self.beta1) * d_w3)
            self.vt_w3 = (self.beta2 * self.vt_w3) + ((1 - self.beta2) * (d_w3 ** 2))
            self.mt_b3 = (self.beta1 * self.mt_b3) + ((1 - self.beta1) * d_b3)
            self.vt_b3 = (self.beta2 * self.vt_b3) + ((1 - self.beta2) * (d_b3 ** 2))

            self.mt_w3 = self.mt_w3 / (1 - self.beta1)
            self.vt_w3 = self.vt_w3 / (1 - self.beta2)
            self.mt_b3 = self.mt_b3 / (1 - self.beta1)
            self.vt_b3 = self.vt_b3 / (1 - self.beta2)

            self.counts += 1
            self.beta1 = 2 / (self.counts + 1)
            self.beta2 = 2 / (self.counts + 1)

            self.w1 -= (self.LEARNING_RATE / np.sqrt(self.vt_w1 + self.eps)) * self.mt_w1
            self.w2 -= (self.LEARNING_RATE / np.sqrt(self.vt_w2 + self.eps)) * self.mt_w2
            self.w3 -= (self.LEARNING_RATE / np.sqrt(self.vt_w3 + self.eps)) * self.mt_w3
            self.b1 -= (self.LEARNING_RATE / np.sqrt(self.vt_b1 + self.eps)) * self.mt_w1
            self.b2 -= (self.LEARNING_RATE / np.sqrt(self.vt_b2 + self.eps)) * self.mt_w2
            self.b3 -= (self.LEARNING_RATE / np.sqrt(self.vt_b3 + self.eps)) * self.mt_w3

    def train(self, input, output):
        # TODO: feed-forward term.
        y1, y2, y3 = self.feedForward(input)
        # TODO: back-propagation term.
        dW1, dW2, dW3, db1, db2, db3 = self.backpropagation(input, output, y1, y2, y3)
        self.update_weight(dW1, dW2, dW3, db1, db2, db3)

    def predict(self, input):
        output = self.feedForward(input)
        return output[2]

    def getAccuracyAndLoss(self, output_of_model, output):
        accuracy = np.mean(np.equal(np.argmax(output_of_model, axis=1), np.argmax(output, axis=1)))

        if self.LOSS == LOSS_CROSSENTROPY:
            # cross entropy loss
            loss = -np.mean(output * np.log(output_of_model) + (1 - output) * np.log(1 - output_of_model))
        elif self.LOSS == LOSS_MSE:
            # MSE
            loss = (1 / 2) * np.mean(((output_of_model - output) ** 2))
        else:
            loss = None

        return accuracy, loss
