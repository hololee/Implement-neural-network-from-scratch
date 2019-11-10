import matplotlib.pyplot as plt
import numpy as np


# plotting func.
def plotting(title, train_acc, train_err, valdiate_acc, validate_err):
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
    ax2.set_ylim([0, 0.3])
    ax2.plot(train_err, "r", label='train')
    ax2.plot(validate_err, "g", label='test')
    ax2.legend(loc='upper right')

    fig.suptitle('learning_rate :{}'.format(title))
    plt.show()
