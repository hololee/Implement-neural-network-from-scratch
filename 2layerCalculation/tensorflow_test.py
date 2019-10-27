import numpy as np
import tensorflow as tf
import os
from nn.datas import DataManager as data_manager

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dataManager = data_manager()


def getAccuracyAndLoss(output_of_model, output):
    accuracy = np.mean(np.equal(np.argmax(output_of_model, axis=1),
                                np.argmax(output, axis=1)))
    loss = np.mean(((output_of_model - output) ** 2))

    return accuracy, loss


def layer(input, outdim):
    w = tf.Variable(tf.random_normal(shape=[input.get_shape().as_list()[-1], outdim]))
    b = tf.Variable(tf.random_normal(shape=[outdim]))

    out = tf.matmul(input, w) + b
    # activation.
    out = tf.nn.relu(out)

    return out


x_input = tf.placeholder(tf.float32, shape=[None, 784])
y_output = tf.placeholder(tf.float32, shape=[None, 10])

# model.
model = layer(x_input, 256)
model = layer(model, 256)
output = layer(model, 10)

# learning_rate = 2e-4
learning_rate_list = [3e-7]
divide = [1]
iter = 500

loss = (1 / 2) * tf.reduce_mean(tf.square(y_output - output))

for learning_rate in learning_rate_list:
    for div in divide:
        rate = learning_rate / div
        print("log.learnin_rate = {:.5}".format(rate))
        optimizing = tf.train.MomentumOptimizer(learning_rate=rate, momentum=0.9).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for iteration in range(iter):
                batch_x, batch_y = dataManager.next_batch(1000)
                _ = sess.run(optimizing, feed_dict={x_input: batch_x, y_output: batch_y})

                y_train_predict = sess.run(output, feed_dict={x_input: dataManager.X_train})
                y_test_predict = sess.run(output, feed_dict={x_input: dataManager.X_test})

                train_acc, train_loss = getAccuracyAndLoss(y_train_predict, dataManager.y_train)
                test_acc, test_loss = getAccuracyAndLoss(y_test_predict, dataManager.y_test)

                print("log.train accuracy : {:.3}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(
                    train_acc,
                    train_loss,
                    test_acc,
                    test_loss))
