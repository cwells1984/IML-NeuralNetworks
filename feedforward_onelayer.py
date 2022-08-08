import copy

import numpy as np

import eval


def sigmoid_function(o):
    r = copy.deepcopy(o)

    for i in range(len(o)):
        r[i] = 1 / (1 + np.exp(-1 * o[i]))

    return r


class FeedForwardNetwork:

    def __init__(self, learn_rate=0.5, num_hidden1=3, type="Classifier"):
        self.learn_rate = learn_rate
        self.num_columns = 0
        self.num_hidden1 = num_hidden1
        self.weights_input_hidden1 = None
        self.weights_hidden1_output = None
        self.type = type

    def fit(self, df, label_columns):

        # Initialize the weights
        self.num_columns = len(df.columns) - len(label_columns)
        self.num_outputs = len(label_columns)
        self.weights_input_hidden1 = np.random.uniform(size=(self.num_columns, self.num_hidden1), low=-.01, high=.01)
        self.weights_hidden1_output = np.random.uniform(size=(self.num_hidden1, self.num_outputs), low=-.01, high=.01)

        optimal_score_reached = False
        if self.type == 'Classifier':
            last_score = -1
        else:
            last_score = np.inf
        while optimal_score_reached == False:

            # After shuffling the rows in the dataframe, extract the matrix of input data X and output data y
            df_X_trn = copy.deepcopy(df)
            df_X_trn = df_X_trn.sample(frac=1)

            y_trn = df_X_trn[label_columns].values

            for label_column in label_columns:
                df_X_trn = df_X_trn.loc[:, df_X_trn.columns != label_column]
            X_trn = df_X_trn.values

            # Modify the weights here
            self.forward_and_back_propagate(X_trn, y_trn, verbose=False)

            # Now check the performance
            y_perf = []
            for index in range(len(X_trn)):
                net_h1, out_h1, net_o, out_o = self.network_forward(X_trn[index])
                y_perf += [net_o]

            # Calculate score
            if (self.type == 'Classifier'):
                score = eval.eval_softmax(y_trn, y_perf)
                if score > last_score:
                    last_score = score
                else:
                    optimal_score_reached = True
            else:
                score = eval.eval_mse(y_trn, y_perf)[0]
                if score < last_score:
                    last_score = score
                else:
                    optimal_score_reached = True

    def predict(self, df, label_columns):

        df_X_tst = copy.deepcopy(df)
        y_truth = df_X_tst[label_columns].values
        for label_column in label_columns:
            df_X_tst = df_X_tst.loc[:, df_X_tst.columns != label_column]
        X_tst = df_X_tst.values

        y_pred = []
        for index in range(len(X_tst)):
            net_h1, out_h1, net_o, out_o = self.network_forward(X_tst[index])
            y_pred += [net_o]

        return y_pred, y_truth

    def fit_predict(self, df_trn, df_tst, label_columns):
        self.fit(df_trn, label_columns)
        y_pred, y_truth = self.predict(df_tst, label_columns)
        return y_pred, y_truth

    # Propagates an input matrix from the Input through the hidden layers, then to the output
    def network_forward(self, X):
        net_h1 = np.array(np.dot(X, self.weights_input_hidden1))
        out_h1 = sigmoid_function(net_h1)
        net_o = np.dot(out_h1, self.weights_hidden1_output)
        out_o = sigmoid_function(net_o)

        return net_h1, out_h1, net_o, out_o

    # Pseudocode pg294
    # https://blog.yani.ai/deltarule/
    def forward_and_back_propagate(self, X_trn, y_trn, verbose=False):

        # For each sample from the training data
        for index in range(len(X_trn)):

            # Calculate updated weights in a new matrix - we want the originals for further back-propagation
            delta_weights_input_hidden1 = np.zeros((self.num_columns, self.num_hidden1))
            delta_weights_hidden1_output = np.zeros((self.num_hidden1, self.num_outputs))

            # First forward thru the network
            net_h1, out_h1, net_o, out_o = self.network_forward(X_trn[index])

            # Back Propagate Output -> V weights
            deriv_and_error = (y_trn[index] - out_o) * out_o * (1 - out_o)
            for h in range(self.num_hidden1):
                for o in range(self.num_outputs):
                    delta_weights_hidden1_output[h][o] = -1 * deriv_and_error[o] * out_h1[h]

            # Back propagate Hidden Layer 1 -> Input
            error_h1 = np.zeros(self.num_hidden1)
            for h in range(len(error_h1)):
                error_h1[h] = np.dot(deriv_and_error, self.weights_hidden1_output[h, :])
            deriv_out_h1 = (out_h1 * (1 - out_h1))
            for i in range(self.num_columns):
                for h in range(self.num_hidden1):
                    delta_weights_input_hidden1[i][h] = -1 * error_h1[h] * deriv_out_h1[h] * X_trn[index, i]

            # New backpropagation code
            #deriv_h1 = out_h1 * (1 - out_h1)
            #error_out = out_o - y_trn[index]
            #error_h1 = np.dot(error_out, self.weights_hidden1_output.T) * deriv_h1
            #delta_weights_hidden1_output = np.dot(np.array([out_h1]).T, np.array([error_out]))
            #delta_weights_input_hidden1 = np.dot(np.array([X_trn[index]]).T, np.array([error_h1]))

            # Now that the entire training set is run through, update
            self.weights_input_hidden1 = self.weights_input_hidden1 - self.learn_rate * delta_weights_input_hidden1
            self.weights_hidden1_output = self.weights_hidden1_output - self.learn_rate * delta_weights_hidden1_output
