import copy

import numpy as np
import pandas as pd

import eval


def sigmoid_function(o):

    r = copy.deepcopy(o)
    for i in range(np.shape(o)[0]):
        for j in range(np.shape(o)[1]):
            r[i][j] = 1 / (1 + np.exp(-1 * o[i][j]))

    return r


class AutoEncodedClassifier:

    def __init__(self, learn_rate=0.5, num_encoding=3, num_hidden1=3):
        self.learn_rate = learn_rate
        self.num_columns = 0
        self.num_encoding = num_encoding
        self.num_hidden1 = num_hidden1
        self.weights_input_encoding = None
        self.weights_encoding_decoding = None
        self.weights_encoding_hidden1 = None
        self.weights_hidden1_output = None

    def fit(self, df, label_columns):

        # Initialize the weights
        self.num_columns = len(df.columns) - len(label_columns)
        self.num_outputs = len(label_columns)
        self.weights_input_encoding = np.random.uniform(size=(self.num_columns, self.num_encoding), low=-.01, high=.01)
        self.weights_encoding_decoding = np.random.uniform(size=(self.num_encoding, self.num_columns), low=-.01,
                                                           high=.01)
        self.weights_encoding_hidden1 = np.random.uniform(size=(self.num_encoding, self.num_hidden1), low=-.01,
                                                          high=.01)
        self.weights_hidden1_output = np.random.uniform(size=(self.num_hidden1, self.num_outputs), low=-.01, high=.01)

        # First AutoEncoding...
        optimal_distance_reached = False
        last_distance = np.inf
        while optimal_distance_reached == False:

            # After shuffling the rows in the dataframe, extract the matrix of input data X and output data y
            df_X_trn = copy.deepcopy(df)
            df_X_trn = df_X_trn.sample(frac=1)

            y_trn = df_X_trn[label_columns].values

            for label_column in label_columns:
                df_X_trn = df_X_trn.loc[:, df_X_trn.columns != label_column]
            X_trn = df_X_trn.values

            # Modify the weights here
            self.autoencode_propagate(X_trn, X_trn, verbose=False)

            # Now check the performance
            encoded_out = np.dot(X_trn, self.weights_input_encoding)
            out = np.dot(encoded_out, self.weights_encoding_decoding)
            mean_distances = np.mean(eval.eval_distance(X_trn, out))

            if mean_distances < last_distance:
                last_distance = mean_distances
            else:
                optimal_distance_reached = True

        # now with the encoded_out of the encoding layer and its corresponding labels, run through a feed forward network
        optimal_score_reached = False
        last_score = -1
        while optimal_score_reached == False:

            # Modify the weights here
            self.forward_and_back_propagate(encoded_out, y_trn, verbose=False)

            # Now check the performance
            h1 = np.dot(encoded_out, self.weights_encoding_hidden1)
            out = np.dot(h1, self.weights_hidden1_output)

            score = eval.eval_softmax(y_trn, out)

            if score > last_score:
                last_score = score
            else:
                optimal_score_reached = True

    def predict(self, df, label_columns):

        df_X_tst = copy.deepcopy(df)
        y_truth = df_X_tst[label_columns].values
        for label_column in label_columns:
            df_X_tst = df_X_tst.loc[:, df_X_tst.columns != label_column]
        X_tst = df_X_tst.values

        h1 = np.dot(X_tst, self.weights_input_encoding)
        h2 = np.dot(h1, self.weights_encoding_hidden1)
        y_pred = np.dot(h2, self.weights_hidden1_output)
        return y_pred, y_truth

    def fit_predict(self, df_trn, df_tst, label_columns):
        self.fit(df_trn, label_columns)
        y_pred, y_truth = self.predict(df_tst, label_columns)
        return y_pred, y_truth

        # Pseudocode pg294
        # https://blog.yani.ai/deltarule/

    def autoencode_propagate(self, X_trn, y_trn, verbose=False):

        # For each sample from the training data
        for index in range(len(X_trn)):

            # Calculate updated weights in a new matrix - we want the originals for further back-propagation
            delta_weights_input_encoding = np.zeros((self.num_columns, self.num_encoding))
            delta_weights_encoding_decoding = np.zeros((self.num_encoding, self.num_columns))

            # Forward X -> Encoding
            net_h1 = np.dot(self.weights_input_encoding.T, X_trn[index].ravel())
            out_h1 = sigmoid_function(net_h1)

            # Encoding -> Decoding
            net_o = np.dot(self.weights_encoding_decoding.T, out_h1)
            out_o = sigmoid_function(net_o)

            # Back Propagate Decoding -> Encoding weights
            deriv_and_error = (y_trn[index] - out_o) * out_o * (1 - out_o)
            for h in range(self.num_encoding):
                for o in range(self.num_columns):
                    delta_weights_encoding_decoding[h][o] = -1 * deriv_and_error[o] * out_h1[h]

            # Back propagate Encoding -> Input weights
            error_h1 = np.zeros(self.num_encoding)
            for h in range(len(error_h1)):
                error_h1[h] = np.dot(deriv_and_error, self.weights_encoding_decoding[h, :])
            deriv_out_h1 = (out_h1 * (1 - out_h1))
            for i in range(self.num_columns):
                for h in range(self.num_encoding):
                    delta_weights_input_encoding[i][h] = -1 * error_h1[h] * deriv_out_h1[h] * X_trn[index, i]

            # Now that the entire training set is run through, update
            self.weights_input_encoding -= self.learn_rate * delta_weights_input_encoding
            self.weights_encoding_decoding -= self.learn_rate * delta_weights_encoding_decoding

    def forward_and_back_propagate(self, X_trn, y_trn, verbose=False):

        # For each sample from the training data
        for index in range(len(X_trn)):

            # Calculate updated weights in a new matrix - we want the originals for further back-propagation
            delta_weights_encoding_hidden1 = np.zeros((self.num_encoding, self.num_hidden1))
            delta_weights_hidden1_output = np.zeros((self.num_hidden1, self.num_outputs))

            # Encoding -> Hidden Layer 1
            net_h1 = np.dot(self.weights_encoding_hidden1.T, X_trn[index].ravel())
            out_h1 = sigmoid_function(net_h1)

            # Hidden Layer 1 -> Output
            net_o = np.dot(self.weights_hidden1_output.T, out_h1)
            out_o = sigmoid_function(net_o)

            # Back Propagate Output -> Hidden Layer 1
            deriv_and_error = (y_trn[index] - out_o) * out_o * (1 - out_o)
            for h in range(self.num_hidden1):
                for o in range(self.num_outputs):
                    delta_weights_hidden1_output[h][o] = -1 * deriv_and_error[o] * out_h1[h]

            # Back propagate Hidden Layer 1 -> Encoding
            error_h1 = np.zeros(self.num_hidden1)
            for h in range(len(error_h1)):
                error_h1[h] = np.dot(deriv_and_error, self.weights_hidden1_output[h, :])
            deriv_out_h1 = (out_h1 * (1 - out_h1))
            for i in range(self.num_encoding):
                for h in range(self.num_hidden1):
                    delta_weights_encoding_hidden1[i][h] = -1 * error_h1[h] * deriv_out_h1[h] * X_trn[index, i]

            # Now that the entire training set is run through, update
            self.weights_encoding_hidden1 -= self.learn_rate * delta_weights_encoding_hidden1
            self.weights_hidden1_output -= self.learn_rate * delta_weights_hidden1_output


class AutoEncodedRegressor:

    def __init__(self, encode_learn_rate=0.5, learn_rate=0.5, num_encoding=3, num_hidden1=3):
        self.encode_learn_rate = encode_learn_rate
        self.learn_rate = learn_rate
        self.num_columns = 0
        self.num_encoding = num_encoding
        self.num_hidden1 = num_hidden1
        self.weights_input_encoding = None
        self.weights_encoding_decoding = None
        self.weights_encoding_hidden1 = None
        self.weights_hidden1_output = None

    def fit(self, df, label_columns):

        # Initialize the weights
        self.num_columns = len(df.columns) - len(label_columns)
        self.num_outputs = len(label_columns)
        self.weights_input_encoding = np.random.uniform(size=(self.num_columns, self.num_encoding), low=-.01, high=.01)
        self.weights_encoding_decoding = np.random.uniform(size=(self.num_encoding, self.num_columns), low=-.01, high=.01)
        self.weights_encoding_hidden1 = np.random.uniform(size=(self.num_encoding, self.num_hidden1), low=-.01, high=.01)
        self.weights_hidden1_output = np.random.uniform(size=(self.num_hidden1, self.num_outputs), low=-.01, high=.01)

        # First AutoEncoding...
        optimal_distance_reached = False
        last_distance = np.inf
        while optimal_distance_reached == False:

            # After shuffling the rows in the dataframe, extract the matrix of input data X and output data y
            df_X_trn = copy.deepcopy(df)
            df_X_trn = df_X_trn.sample(frac=1)

            y_trn = df_X_trn[label_columns].values

            for label_column in label_columns:
                df_X_trn = df_X_trn.loc[:, df_X_trn.columns != label_column]
            X_trn = df_X_trn.values

            # Modify the weights here
            self.autoencode_propagate(X_trn, X_trn, verbose=False)

            # Now check the performance
            encoded_out = []
            decoded_out = []
            for index in range(len(X_trn)):
                X_index = np.array([X_trn[index]])
                net_h1 = np.array(np.dot(X_index, self.weights_input_encoding))
                out_h1 = sigmoid_function(net_h1)
                encoded_out += [out_h1[0]]
                net_o = np.dot(out_h1, self.weights_encoding_decoding)
                out_o = sigmoid_function(net_o)
                decoded_out += [out_o[0]]

            mean_distances = np.mean(eval.eval_distance(X_trn, decoded_out))

            if mean_distances < last_distance:
                last_distance = mean_distances
            else:
                print(f"loss= {mean_distances}")
                print(f"typ distance= {eval.eval_distance([X_trn[0]], [X_trn[-1]])}")
                optimal_distance_reached = True

        # now with the encoded_out of the encoding layer and its corresponding labels, run through a feed forward network
        optimal_mse_reached = False
        last_mse = np.inf
        while optimal_mse_reached == False:

            # Modify the weights here
            self.forward_and_back_propagate(np.array(encoded_out), y_trn, verbose=False)

            # Now check the performance
            h1 = np.dot(encoded_out, self.weights_encoding_hidden1)
            out = np.dot(h1, self.weights_hidden1_output)

            mse = eval.eval_mse(y_trn, out)[0]

            if mse < last_mse:
                last_mse = mse
            else:
                optimal_mse_reached = True

    def predict(self, df, label_columns):

        df_X_tst = copy.deepcopy(df)
        y_truth = df_X_tst[label_columns].values
        for label_column in label_columns:
            df_X_tst = df_X_tst.loc[:, df_X_tst.columns != label_column]
        X_tst = df_X_tst.values

        y_pred = []
        for index in range(len(X_tst)):
            net_h1 = np.dot(X_tst[index], self.weights_input_encoding)
            out_h1 = net_h1
            net_o = np.dot(out_h1, self.weights_encoding_decoding)
            out_o = net_o.item((0,))
            y_pred += [out_o]

        return y_pred, y_truth

    def fit_predict(self, df_trn, df_tst, label_columns):
        self.fit(df_trn, label_columns)
        y_pred, y_truth = self.predict(df_tst, label_columns)
        return y_pred, y_truth

    # Pseudocode pg294
    # https://blog.yani.ai/deltarule/
    def autoencode_propagate(self, X_trn, y_trn, verbose=False):
        #print(np.shape(X_trn))

        # For each sample from the training data
        for index in range(len(X_trn)):

            X_index = np.array([X_trn[index]])
            y_index = y_trn[index]

            # Forward X -> Encoding
            net_h1 = np.array(np.dot(X_index, self.weights_input_encoding))
            out_h1 = sigmoid_function(net_h1)

            # Encoding -> Decoding
            net_o = np.dot(out_h1, self.weights_encoding_decoding)
            out_o = sigmoid_function(net_o)

            # Now Back-propagate
            deriv_h1 = out_h1 * (1 - out_h1)
            error = out_o - y_index
            delta_h1 = np.multiply(np.dot(error, self.weights_encoding_decoding.T), deriv_h1)
            delta_weights_encoding_decoding = np.dot(out_h1.T, error)
            delta_weights_input_encoding = np.dot(X_index.T, delta_h1)

            # Now that the entire training set is run through, update
            self.weights_input_encoding = self.weights_input_encoding - (self.encode_learn_rate * delta_weights_input_encoding)
            self.weights_encoding_decoding = self.weights_encoding_decoding - (self.encode_learn_rate * delta_weights_encoding_decoding)

    def forward_and_back_propagate(self, X_trn, y_trn, verbose=False):

        # For each sample from the training data
        for index in range(len(X_trn)):

            X_index = np.array([X_trn[index]])
            y_index = y_trn[index]

            # Encoding -> Hidden Layer 1
            net_h1 = np.array(np.dot(X_index, self.weights_encoding_hidden1))
            out_h1 = sigmoid_function(net_h1)

            # Hidden Layer 1 -> Output
            net_o = np.dot(out_h1, self.weights_hidden1_output)
            out_o = sigmoid_function(net_o)

            # Now Back-propagate
            deriv_h1 = out_h1 * (1 - out_h1)
            error = out_o - y_trn[index]
            delta_h1 = np.multiply(np.dot(error, self.weights_hidden1_output.T), deriv_h1)
            delta_weights_hidden1_output = np.dot(out_h1.T, error)
            delta_weights_encoding_hidden1 = np.dot(X_index.T, delta_h1)

            # Now that the entire training set is run through, update
            self.weights_encoding_hidden1 = self.weights_encoding_hidden1 - (self.learn_rate * delta_weights_encoding_hidden1)
            self.weights_hidden1_output = self.weights_hidden1_output - (self.learn_rate * delta_weights_hidden1_output)
