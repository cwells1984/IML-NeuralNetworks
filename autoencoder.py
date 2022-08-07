import copy

import numpy as np
import pandas as pd

import eval


# Applies the sigmoid function to each element in the input matrix
def sigmoid_function(o):

    r = copy.deepcopy(o)
    for i in range(np.shape(o)[0]):
        for j in range(np.shape(o)[1]):
            r[i][j] = 1 / (1 + np.exp(-1 * o[i][j]))

    return r


class AutoEncodedNetwork:

    def __init__(self, encode_learn_rate=0.5, learn_rate=0.5, num_encoding=3, num_hidden1=3, type="Classifier"):
        self.encode_learn_rate = encode_learn_rate
        self.learn_rate = learn_rate
        self.num_columns = 0
        self.num_encoding = num_encoding
        self.num_hidden1 = num_hidden1
        self.weights_input_encoding = None
        self.weights_encoding_decoding = None
        self.weights_encoding_hidden1 = None
        self.weights_hidden1_output = None
        self.type = type
        self.min_loss = 0.01

    # Propagates an input matrix X through the Encoding layer, then the decoding layer
    def autoencoder_forward(self, X):
        net_h1 = np.array(np.dot(X, self.weights_input_encoding))
        out_h1 = sigmoid_function(net_h1)
        net_o = np.dot(out_h1, self.weights_encoding_decoding)
        out_o = sigmoid_function(net_o)

        return net_h1, out_h1, net_o, out_o

    # Propagates an input matrix from the Encoding layer through the Hidden Layer, then to the output
    def network_forward(self, X):
        net_h1 = np.array(np.dot(X, self.weights_encoding_hidden1))
        out_h1 = sigmoid_function(net_h1)
        net_o = np.dot(out_h1, self.weights_hidden1_output)
        out_o = sigmoid_function(net_o)

        return net_h1, out_h1, net_o, out_o

    # Sets the network weights based on the training set
    def fit(self, df, label_columns):

        # Initialize the weights
        self.num_columns = len(df.columns) - len(label_columns)
        self.num_outputs = len(label_columns)
        self.weights_input_encoding = np.random.uniform(size=(self.num_columns, self.num_encoding), low=-.01, high=.01)
        self.weights_encoding_decoding = np.random.uniform(size=(self.num_encoding, self.num_columns), low=-.01, high=.01)
        self.weights_encoding_hidden1 = np.random.uniform(size=(self.num_encoding, self.num_hidden1), low=-.01, high=.01)
        self.weights_hidden1_output = np.random.uniform(size=(self.num_hidden1, self.num_outputs), low=-.01, high=.01)

        # First, auto-encode until the loss reaches its minimum value
        optimal_distance_reached = False
        last_distance = np.inf
        while optimal_distance_reached == False:

            # After shuffling the rows in the dataframe, extract the matrix of input data X and output data y
            df_X_trn = copy.deepcopy(df)
            df_X_trn = df_X_trn.sample(frac=1)

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
                net_h1, out_h1, net_o, out_o = self.autoencoder_forward(X_index)
                encoded_out += [out_h1[0]]
                decoded_out += [out_o[0]]

            # Get the mean of all the distances of the decoded features to the training features
            mean_distances = np.mean(eval.eval_distance(X_trn, decoded_out))

            if mean_distances < last_distance and mean_distances > self.min_loss:
                last_distance = mean_distances
            else:
                print(f"Autoencoded with mean loss= {mean_distances:.2f}")
                optimal_distance_reached = True

        # now with the encoded_out of the encoding layer and its corresponding labels, run through a feed forward network
        optimal_score_reached = False
        if self.type == 'Classifier':
            last_score = -1
        else:
            last_score = np.inf

        while optimal_score_reached == False:

            # After shuffling the rows in the dataframe, extract the matrix of input data X and output data y
            df_X_trn = copy.deepcopy(df)
            df_X_trn = df_X_trn.sample(frac=1)
            y_trn = df[label_columns].values

            for label_column in label_columns:
                df_X_trn = df_X_trn.loc[:, df_X_trn.columns != label_column]
            X_trn = df_X_trn.values

            # Get the encoded values of X
            net_encoded, out_encoded, net_decoded, out_decoded = self.autoencoder_forward(X_trn)

            # Modify the weights here
            self.forward_and_back_propagate(np.array(net_encoded), y_trn, verbose=False)

            # Now check the performance
            net_h1, out_h1, net_o, out_o = self.network_forward(net_encoded)

            # Calculate score
            if (self.type == 'Classifier'):
                score = eval.eval_softmax(y_trn, net_o)
                if score > last_score:
                    last_score = score
                else:
                    optimal_score_reached = True
            else:
                score = eval.eval_mse(y_trn, net_o)[0]

                if score < last_score:
                    last_score = score
                else:
                    optimal_score_reached = True

    # Using the existing weights, make predictions with a testing set
    def predict(self, df, label_columns):

        # Create a matrix of values for the features in the test set and the labels
        df_X_tst = copy.deepcopy(df)
        y_truth = df_X_tst[label_columns].values
        for label_column in label_columns:
            df_X_tst = df_X_tst.loc[:, df_X_tst.columns != label_column]
        X_tst = df_X_tst.values

        # Now feed them through the network to generate the predictions
        y_pred = []
        for index in range(len(X_tst)):
            X_index = np.array([X_tst[index]])
            out_h1 = np.array(np.dot(X_index, self.weights_input_encoding))
            out_o = np.dot(out_h1, self.weights_encoding_decoding)
            y_pred += [out_o[0]]

        return y_pred, y_truth

    # Sets the weights based on the training set and makes predictions based on the test set
    def fit_predict(self, df_trn, df_tst, label_columns):
        self.fit(df_trn, label_columns)
        y_pred, y_truth = self.predict(df_tst, label_columns)
        return y_pred, y_truth

    # Conducts the forward and back propagation thru the autoencoder
    def autoencode_propagate(self, X_trn, y_trn, verbose=False):

        # For each sample from the training data
        for index in range(len(X_trn)):

            X_index = np.array([X_trn[index]])
            y_index = y_trn[index]

            # First forward thru the network
            net_h1, out_h1, net_o, out_o = self.autoencoder_forward(X_index)

            # Now Back-propagate
            deriv_h1 = out_h1 * (1 - out_h1)
            error = out_o - y_index
            delta_h1 = np.multiply(np.dot(error, self.weights_encoding_decoding.T), deriv_h1)
            delta_weights_encoding_decoding = np.dot(out_h1.T, error)
            delta_weights_input_encoding = np.dot(X_index.T, delta_h1)

            # Now that the entire training set is run through, update
            self.weights_input_encoding = self.weights_input_encoding - (self.encode_learn_rate * delta_weights_input_encoding)
            self.weights_encoding_decoding = self.weights_encoding_decoding - (self.encode_learn_rate * delta_weights_encoding_decoding)

    # Conducts the forward and back propagation thru the 1-layer neural network fed by the encoder
    def forward_and_back_propagate(self, X_trn, y_trn, verbose=False):

        # For each sample from the training data
        for index in range(len(X_trn)):

            X_index = np.array([X_trn[index]])
            y_index = y_trn[index]

            # First forward thru the network
            net_h1, out_h1, net_o, out_o = self.network_forward(X_index)

            # Now Back-propagate
            deriv_h1 = out_h1 * (1 - out_h1)
            error = out_o - y_index
            delta_h1 = np.multiply(np.dot(error, self.weights_hidden1_output.T), deriv_h1)
            delta_weights_hidden1_output = np.dot(out_h1.T, error)
            delta_weights_encoding_hidden1 = np.dot(X_index.T, delta_h1)

            # Now that the entire training set is run through, update
            self.weights_encoding_hidden1 = self.weights_encoding_hidden1 - (self.learn_rate * delta_weights_encoding_hidden1)
            self.weights_hidden1_output = self.weights_hidden1_output - (self.learn_rate * delta_weights_hidden1_output)
