import numpy as np
# my class
import nn.model
import nn.tools as tool
from nn.model import NeuralNetwork as network
from nn.datas import DataManager as data_manager

# data stack for plotting.
train_acc = []
train_err = []
valid_acc = []
valid_err = []

# config list.
INFO_SIGMOID_MOMENTUM_MSE_BATCH = {'total_epoch': 1000,
                                   'batch_size': 60000,
                                   'learning_rate': 1e-6,
                                   'random_seed': 42,
                                   'train_dataset_size': 60000,
                                   'test_dataset_size': 10000,
                                   'momentum': 0.8,
                                   'optimizer': nn.model.OPTIMIZER_GD_MOMENTUM,
                                   'activation': nn.model.ACTIVATE_SIGMOID,
                                   'loss': nn.model.LOSS_MSE}

#3e-8
INFO_RELU_GD_MSE_BATCH = {'total_epoch': 100,
                          'batch_size': 60000,
                          'learning_rate': 1e-6,
                          'random_seed': 42,
                          'train_dataset_size': 60000,
                          'test_dataset_size': 10000,
                          'momentum': 0.8,
                          'optimizer': nn.model.OPTIMIZER_GD_MOMENTUM,
                          'activation': nn.model.ACTIVATE_RELU,
                          'loss': nn.model.LOSS_MSE}

# set config.
current_config = INFO_RELU_GD_MSE_BATCH

#
# config_assignmentA_MOMENTUM_CROSSENTROPY = {'total_epoch': 50, 'batch_size': 1000, 'learning_rate': 0.1,
#                                             'random_seed': 42,
#                                             'train_dataset_size': 60000, 'test_dataset_size': 10000, 'momentum': 0.9,
#                                             'optimizer': nn.model.OPTIMIZER_GD_MOMENTUM,
#                                             'activation': nn.model.ACTIVATE_SIGMOID, 'loss': nn.model.LOSS_CROSSENTROPY}
#
# config_assignmentA_CROSSENTROPY = {'total_epoch': 50, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
#                                    'train_dataset_size': 60000, 'test_dataset_size': 10000,
#                                    'optimizer': nn.model.OPTIMIZER_GD,
#                                    'activation': nn.model.ACTIVATE_SIGMOID, 'loss': nn.model.LOSS_CROSSENTROPY}
#
# config_assignmentB_CROSSENTROPY = {'total_epoch': 50, 'batch_size': 1000, 'learning_rate': 0.1, 'random_seed': 42,
#                                    'train_dataset_size': 60000, 'test_dataset_size': 10000,
#                                    'optimizer': nn.model.OPTIMIZER_GD,
#                                    'activation': nn.model.ACTIVATE_RELU, 'loss': nn.model.LOSS_CROSSENTROPY}
#
# config_assignmentC_MINI_BATCH_CROSSENTROPY = {'total_epoch': 50, 'batch_size': 1000, 'learning_rate': 0.1,
#                                               'random_seed': 42,
#                                               'train_dataset_size': 60000, 'test_dataset_size': 10000,
#                                               'optimizer': nn.model.OPTIMIZER_GD,
#                                               'activation': nn.model.ACTIVATE_RELU, 'loss': nn.model.LOSS_CROSSENTROPY}
#
# config_assignmentC_BATCH_CROSSENTROPY = {'total_epoch': 80, 'batch_size': 60000, 'learning_rate': 0.1,
#                                          'random_seed': 42,
#                                          'train_dataset_size': 60000, 'test_dataset_size': 10000,
#                                          'optimizer': nn.model.OPTIMIZER_GD,
#                                          'activation': nn.model.ACTIVATE_RELU, 'loss': nn.model.LOSS_CROSSENTROPY}
#
# config_assignmentC_STOCHASTIC_CROSSENTROPY = {'total_epoch': 10, 'batch_size': 1, 'learning_rate': 0.01,
#                                               'random_seed': 42,
#                                               'train_dataset_size': 60000, 'test_dataset_size': 10000,
#                                               'optimizer': nn.model.OPTIMIZER_GD,
#                                               'activation': nn.model.ACTIVATE_RELU, 'loss': nn.model.LOSS_CROSSENTROPY}
#
# config_assignmentD_ADAGRAD_CROSSENTROPY = {'total_epoch': 30, 'batch_size': 1000, 'learning_rate': 0.005,
#                                            'random_seed': 42,
#                                            'train_dataset_size': 60000, 'test_dataset_size': 10000,
#                                            'optimizer': nn.model.OPTIMIZER_ADAGRAD,
#                                            'activation': nn.model.ACTIVATE_RELU, 'loss': nn.model.LOSS_CROSSENTROPY,
#                                            'epsilon': 1e-5}
#
# config_assignmentD_ADAM_CROSSENTROPY = {'total_epoch': 30, 'batch_size': 1000, 'learning_rate': 0.005,
#                                         'random_seed': 42,
#                                         'train_dataset_size': 60000, 'test_dataset_size': 10000,
#                                         'optimizer': nn.model.OPTIMIZER_ADAM,
#                                         'activation': nn.model.ACTIVATE_RELU, 'loss': nn.model.LOSS_CROSSENTROPY,
#                                         'beta1': 0.9,
#                                         'beta2': 0.999,
#                                         'epsilon': 1e-8}

# define network nn.
network_model = network(configure=current_config, h1=256, h2=256)
dataManager = data_manager()

# fix the random value.
# np.random.seed(network_model.SEED)

# using mini-batch
for i in range(network_model.TOTAL_EPOCH):
    print("============== EPOCH {} START ==============".format(i + 1))
    for j in range(dataManager.train_dataset_size // network_model.BATCH_SIZE):
        # print("-------------- batch {} training...".format(j))

        # load batch data.
        batch_x, batch_y = dataManager.next_batch(network_model.BATCH_SIZE)

        # train model.
        network_model.train(batch_x, batch_y)

        if current_config["batch_size"] == 1:
            if j % 100 == 0:
                # save data.
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
        else:
            if j % 10 == 0:
                # save data.
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
    print("============== EPOCH {} END ================".format(i + 1))

    # shake data when epoch ended.
    # dataManager.shake_data()

    # calculate accuracy and loss
    output_train = network_model.predict(dataManager.X_train)
    accuracy_train, loss_train = network_model.getAccuracyAndLoss(output_train, dataManager.y_train)

    # calculate test dataset.
    output_test = network_model.predict(dataManager.X_test)
    accuracy_test, loss_test = network_model.getAccuracyAndLoss(output_test, dataManager.y_test)

    print("train accuracy : {:.4}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(accuracy_train, loss_train,
                                                                                             accuracy_test, loss_test))

    if i % 10 == 0 or i == network_model.TOTAL_EPOCH - 1:
        # draw graph.
        tool.plotting(current_config['learning_rate'], train_acc, train_err, valid_acc, valid_err)
