import numpy as np


# Returns the classification score for a specified ground truth - predicted value pair
def eval_classification_score(y_truth, y_pred):
    y_corr = 0
    num_y = len(y_truth)

    # calculates the fraction of correct predictions
    for i in range(num_y):
        if y_truth[i] == y_pred[i]:
            y_corr += 1

    return y_corr / num_y


# Calculates the euclidean distance between 2 matrices, with each row in the matrix a vector
def eval_distance(x, y):
    distances = []
    for i in range(len(x)):
        vector_x = x[i]
        vector_y = y[i]
        sum_squares = 0
        for j in range(len(vector_x)):
            sum_squares += (vector_x[j] - vector_y[j])**2
        distances += [np.power(sum_squares, 0.5)]
    return distances


# Returns the mean squared error for a specified ground truth - predicted value pair
def eval_mse(y_truth, y_pred):
    y_total_error = 0
    num_y = len(y_truth)

    # sums the squares of the errors
    for i in range(num_y):
        y_total_error += (y_pred[i] - y_truth[i]) ** 2

    return (y_total_error / num_y)


def eval_softmax(y_truth, y_pred):
    y_corr = 0
    num_y = len(y_truth)

    for i in range(num_y):
        expected_index = np.argmax(y_truth[i])
        actual_index = np.argmax(softmax(y_pred[i]))
        if expected_index == actual_index:
            y_corr += 1

    return y_corr / num_y


# Returns whether the error for a specified ground truth is within a predicted threshold
def eval_thresh(y_truth, y_pred, thresh):
    y_corr = 0
    num_y = len(y_truth)

    # calculates the fraction of correct predictions
    for i in range(num_y):
        if np.abs(y_truth[i] - y_pred[i]) < thresh:
            y_corr += 1

    return y_corr / num_y

def softmax(z):
    sigma = []

    for i in range(len(z)):
        denominator = 0
        for j in range(len(z)):
            denominator += np.exp(z[j])
        sigma += [np.exp(z[i]) / denominator]

    return sigma
