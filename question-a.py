import numpy as np
import matplotlib.pyplot as plt

# my class
from nn.model import NeuralNetwork
import nn.tools as tool
import nn.datas as DataManager

# data stack for ploting.
train_acc = []
train_err = []
valid_acc = []
valid_err = []


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

    # Network nn class.


# fix the random value.
np.random.seed(NeuralNetwork.SEED)

# set config.
config = {'a': 3}

# define network nn.
network_model = NeuralNetwork(learning_rate=NeuralNetwork.LEARNING_RATE)

# using mini-batch
for i in range(NeuralNetwork.TOTAL_EPOCH):
    print("============== EPOCH {} START ==============".format(i + 1))
    for j in range(NeuralNetwork.TRAIN_DATASET_SIZE // NeuralNetwork.BATCH_SIZE):
        print("-------------- batch {} training...".format(j))
        batch_x, batch_y = network_model.next_batch(NeuralNetwork.BATCH_SIZE)

        # TODO: feed-forward term.
        y1, y2, y3 = network_model.feedForward_sigmoid(batch_x)

        # TODO: back-propagation term.
        dW1, dW2, dW3 = network_model.backpropagation_sigmoid(batch_x, batch_y, y1, y2, y3)
        network_model.update_weight(dW1, dW2, dW3)

    print("============== EPOCH {} END ================".format(i + 1))

    # shake data when epoch ended.
    network_model.shake_data()

    # add one epoch data.
    _, _, y_train3 = network_model.feedForward_sigmoid(network_model.X_train)
    match_prediction_train = np.equal(np.argmax(y_train3, axis=1),
                                      np.argmax(network_model.y_train, axis=1))
    accuracy_train = np.mean(match_prediction_train)
    match_loss_train = (1 / 2) * ((y_train3 - network_model.y_train) ** 2)

    train_acc.append(accuracy_train)
    train_err.append(np.mean(match_loss_train))

    # calculate test dataset.
    _, _, y_test3 = network_model.feedForward_sigmoid(network_model.X_test)
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
