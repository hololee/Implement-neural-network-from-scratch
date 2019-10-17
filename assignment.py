import numpy as np
# my class
import nn.model
from nn.model import NeuralNetwork as network
import nn.tools as tool
from nn.datas import DataManager as data_manager

# data stack for plotting.
train_acc = []
train_err = []
valid_acc = []
valid_err = []

# set config.
config_assignmentA = {'total_epoch': 120, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                      'train_dataset_size': 60000, 'test_dataset_size': 10000, 'optimizer': nn.model.OPTIMIZER_GD,
                      'activation': nn.model.ACTIVATE_SIGMOID}

config_assignmentB = {'total_epoch': 120, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                      'train_dataset_size': 60000, 'test_dataset_size': 10000, 'optimizer': nn.model.OPTIMIZER_GD,
                      'activation': nn.model.ACTIVATE_RELU}

config_assignmentC_MINI_BATCH = {'total_epoch': 120, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                                 'train_dataset_size': 60000, 'test_dataset_size': 10000,
                                 'optimizer': nn.model.OPTIMIZER_GD,
                                 'activation': nn.model.ACTIVATE_RELU}

config_assignmentC_BATCH = {'total_epoch': 120, 'batch_size': 60000, 'learning_rate': 0.1, 'random_seed': 42,
                            'train_dataset_size': 60000, 'test_dataset_size': 10000,
                            'optimizer': nn.model.OPTIMIZER_GD,
                            'activation': nn.model.ACTIVATE_RELU}

config_assignmentC_STOCHASTIC = {'total_epoch': 120, 'batch_size': 1, 'learning_rate': 0.01, 'random_seed': 42,
                                 'train_dataset_size': 60000, 'test_dataset_size': 10000,
                                 'optimizer': nn.model.OPTIMIZER_GD,
                                 'activation': nn.model.ACTIVATE_RELU}

config_assignmentD_ADAGRAD = {'total_epoch': 120, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                              'train_dataset_size': 60000, 'test_dataset_size': 10000,
                              'optimizer': nn.model.OPTIMIZER_ADAGRAD,
                              'activation': nn.model.ACTIVATE_RELU,
                              'epsilon': 1e-5}

config_assignmentD_ADAM = {'total_epoch': 120, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
                           'train_dataset_size': 60000, 'test_dataset_size': 10000,
                           'optimizer': nn.model.OPTIMIZER_ADAM,
                           'activation': nn.model.ACTIVATE_RELU}

# define network nn.
network_model = network(configure=config_assignmentD_ADAGRAD, h1=100, h2=50)
dataManager = data_manager()

# fix the random value.
np.random.seed(network_model.SEED)

# using mini-batch
for i in range(network_model.TOTAL_EPOCH):
    print("============== EPOCH {} START ==============".format(i + 1))
    for j in range(dataManager.train_dataset_size // network_model.BATCH_SIZE):
        if network_model.configure != config_assignmentC_STOCHASTIC:
            print("-------------- batch {} training...".format(j))

        # load batch data.
        batch_x, batch_y = dataManager.next_batch(network_model.BATCH_SIZE)

        # train model.
        network_model.train(batch_x, batch_y)
    print("============== EPOCH {} END ================".format(i + 1))

    # shake data when epoch ended.
    dataManager.shake_data()

    # calculate accuracy and loss
    output_train = network_model.predict(dataManager.X_train)
    accuracy_train, loss_train = network_model.getAccuracyAndLoss(output_train, dataManager.y_train)

    # add data to stack
    train_acc.append(accuracy_train)
    train_err.append(loss_train)

    # calculate test dataset.
    output_test = network_model.predict(dataManager.X_test)
    accuracy_test, loss_test = network_model.getAccuracyAndLoss(output_test, dataManager.y_test)

    # add data to stack
    valid_acc.append(accuracy_test)
    valid_err.append(loss_test)

    print("train accuracy : {:.4}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(accuracy_train, loss_train,
                                                                                             accuracy_test, loss_test))
    # draw graph.
    tool.plotting(train_acc, train_err, valid_acc, valid_err)
