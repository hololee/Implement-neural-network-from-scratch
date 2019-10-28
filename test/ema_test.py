import numpy as np


def exponentioal_moving_average(previous_stacked_weights, previous_stacked_weight_averages, weights):
    # stacked_weights shape = (i , x, y) x, y is weights,  i is stacked data.
    n = len(previous_stacked_weights) + 1

    alpha = 2 / (n + 1)

    if len(previous_stacked_weights) == 1:
        prev_average_values = weights
        alpha = 0.9
    else:
        prev_average_values = previous_stacked_weight_averages

    cur_average_values = (alpha * weights) + (1 - alpha) * prev_average_values
    stacked_weights = np.concatenate((previous_stacked_weights, weights))

    return stacked_weights, cur_average_values


a = np.ones((10, 10)) * 1
a = a[np.newaxis, :]
stacked = a
average = None

for i in range(10):
    stacked, average = exponentioal_moving_average(stacked, average, a * i)
    print(average)
