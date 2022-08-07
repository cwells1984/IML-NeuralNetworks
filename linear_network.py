import copy
import eval
import numpy as np
import preprocessing


def sigmoid_function(o):
    if o < 0.5:
        return 0
    else:
        return 1


def calc_exp(o):
    sum_exp = 0
    for i in range(len(o)):
        sum_exp += np.exp(o[i])

    y = []
    for i in range(len(o)):
        y += [np.exp(o[i]) / sum_exp]

    return y


class LinearMultiClassifier:

    def __init__(self, learn_rate=0.5, reg_value=0):
        self.learn_rate = learn_rate
        self.reg_value = reg_value
        self.weights = None
        self.num_classes = 0
        self.target_classes = []

    def fit(self, df, label_column):
        num_columns = len(df.columns) - 1
        self.num_classes = len(df[label_column].unique())
        #print(f"Logistic regression on {len(df)} entries for {self.num_classes} target classes")

        last_weights = np.zeros((self.num_classes, num_columns))
        self.weights = np.random.uniform(size=(self.num_classes, num_columns), low=-.01, high=.01)
        #print(f"weights before= {self.weights}")
        weights_delta = np.zeros((self.num_classes, num_columns))

        # Make a 2d array of the features
        X = df.loc[:, df.columns != label_column].values
        r_single = df.loc[:, df.columns == label_column].values.ravel()

        # Encode the target values as a matrix (1 indicates it is in the class, 0 not)
        df_r = preprocessing.encode_onehot(df, label_column)
        df_r = df_r.iloc[:, (-1 * self.num_classes):]
        r = df_r.values
        self.target_classes = []
        for col in df_r.columns:
            self.target_classes += [col.split("_")[-1]]

        # Setup for loop
        does_not_converge = True
        last_score = 0

        while does_not_converge:
            y = []
            weights_delta = np.zeros((self.num_classes, num_columns))

            for i in range(0, len(X)):
                #print(f"row {i}")
                o = np.zeros(self.num_classes)

                # for each class calculate o
                for k in range(self.num_classes):
                    for j in range(num_columns):
                        o[k] += self.weights[k][j] * X[i][j]

                # calculate predicted values
                y += [calc_exp(o)]

                # update weights delta
                for k in range(self.num_classes):
                    for j in range(num_columns):
                        weights_delta[k][j] += (r[i][k] - y[-1][k]) * X[i][j]

            # Now update the weights
            last_weights = copy.deepcopy(self.weights)
            for k in range(self.num_classes):
                for j in range(num_columns):
                    self.weights[k][j] += self.learn_rate * weights_delta[k][j]
                    self.weights[k][j] -= self.reg_value * self.weights[k][j]

            # Translate these y values to predictions and score them
            y_pred = []
            for y_max in np.argmax(y, axis=1):
                y_pred += [self.target_classes[y_max]]
            score = eval.eval_classification_score(r_single, y_pred)
            #print(f"acc score= {score * 100:.2f}%")

            if score > last_score:
                does_not_converge = True
                last_score = score
            else:
                does_not_converge = False

        #print(f"Converged! {score*100:.2f}% <= {last_score*100:.2f}%")
        #print(f"weights after= {self.weights}")

    def predict(self, df, label_column):
        #print(f"Logistic regression on {len(df)} entries")
        num_columns = len(df.columns) - 1

        X = df.loc[:, df.columns != label_column].values
        y = []

        for i in range(0, len(X)):
            o = np.zeros(self.num_classes)

            # for each class calculate o
            for k in range(self.num_classes):
                for j in range(num_columns):
                    o[k] += self.weights[k][j] * X[i][j]

            # calculate predicted values
            y += [calc_exp(o)]

            # Translate these y values to predictions and score them
            y_pred = []
            for y_max in np.argmax(y, axis=1):
                y_pred += [self.target_classes[y_max]]

        return y_pred

    def fit_predict(self, df_trn, df_test, label_column):
        self.fit(df_trn, label_column)
        y_pred = self.predict(df_test, label_column)
        return y_pred


class LinearClassifier:

    def __init__(self, learn_rate=0.5, reg_value=0):
        self.learn_rate = learn_rate
        self.reg_value = reg_value
        self.weights = None

    def fit(self, df, label_column):
        #print(f"Logistic regression on {len(df)} entries")
        num_columns = len(df.columns) - 1
        last_weights = np.zeros(num_columns)
        self.weights = np.random.uniform(size=num_columns, low=-.01, high=.01)
        #print(f"weights before= {self.weights}")
        weights_delta = np.zeros(num_columns)

        X = df.loc[:, df.columns != label_column].values
        r = df.loc[:, df.columns == label_column].values.ravel()
        does_not_converge = True
        last_score = 0

        while does_not_converge:
            y = []
            weights_delta = np.zeros(num_columns)

            for i in range(0, len(X)):
                # print(f"row {i}")

                # calculate o to feed into sigmoid and get predicted y
                o = 0
                for j in range(0, num_columns):
                    o += self.weights[j] * X[i][j]
                    # print(f"column {j} o={o}")
                y += [sigmoid_function(o)]
                # print(f"column {j} o={o}, y={y[-1]}")

                # update the weights_delta
                for j in range(0, num_columns):
                    weights_delta[j] += (r[i] - y[-1]) * X[i][j]
                # print(f"weights_delta= {weights_delta}")

                # now update the weights
                last_weights = copy.deepcopy(self.weights)
                for j in range(0, num_columns):
                    self.weights[j] += (self.learn_rate * weights_delta[j])
                    self.weights[j] -= self.reg_value * self.weights[j]

                # print(f"r= {r}")
                # print(f"y= {y}")

            # check for convergence - did the # of misclassifications decrease?
            score = eval.eval_classification_score(r, y)
            #print(f"acc score= {score * 100:.2f}%")

            if score > last_score:
                does_not_converge = True
                last_score = score
            else:
                does_not_converge = False

        #print(f"Converged! {score*100:.2f}% < {last_score*100:.2f}%")
        #print(f"weights after= {self.weights}")

    def predict(self, df, label_column):
        #print(f"Logistic regression on {len(df)} entries")
        num_columns = len(df.columns) - 1

        X = df.loc[:, df.columns != label_column].values
        y_pred = []

        for i in range(len(X)):
            o = 0
            for j in range(0, num_columns):
                o += X[i][j] * self.weights[j]
            y_pred += [sigmoid_function(o)]

        return y_pred

    def fit_predict(self, df_trn, df_test, label_column):
        self.fit(df_trn, label_column)
        y_pred = self.predict(df_test, label_column)
        return y_pred


class LinearRegressor:

    def __init__(self, learn_rate=0.5, reg_value=0):
        self.learn_rate = learn_rate
        self.reg_value = reg_value
        self.weights = None

    def fit(self, df, label_column):
        #print(f"Logistic regression on {len(df)} entries")
        num_columns = len(df.columns) - 1
        last_weights = np.zeros(num_columns)
        self.weights = np.random.uniform(size=num_columns, low=-.01, high=.01)
        weights_delta = np.zeros(num_columns)

        X = df.loc[:, df.columns != label_column].values
        r = df.loc[:, df.columns == label_column].values.ravel()
        does_not_converge = True
        last_mse = np.inf

        while does_not_converge:
            y = []
            weights_delta = np.zeros(num_columns)

            for i in range(0, len(X)):
                #print(f"row {i}")

                # calculate o to feed into sigmoid and get predicted y
                o = 0
                for j in range(0, num_columns):
                    o += self.weights[j] * X[i][j]
                    #print(f"column {j} o={o}")
                y += [o]
                #print(f"column {j} o={o}, y={y[-1]}")

                # update the weights_delta
                for j in range(0, num_columns):
                    weights_delta[j] += (r[i] - y[-1]) * X[i][j]
                #print(f"weights_delta= {weights_delta}")

                # now update the weights
                for j in range(0, num_columns):
                    last_weights = copy.deepcopy(self.weights)
                    self.weights[j] += (self.learn_rate * weights_delta[j])
                    self.weights[j] -= self.reg_value * self.weights[j]

            # check for convergence - did the mse increase?
            #print(f"r= {r}")
            #print(f"y= {y}")
            mse = eval.eval_mse(r, y)
            #print(f"mse= {mse:.2f}")

            if mse <= last_mse:
                does_not_converge = True
                last_mse = mse
            else:
                does_not_converge = False

        #print(f"Converged! {mse} > {last_mse}")
        #self.weights = last_weights

    def predict(self, df, label_column):
        #print(f"Logistic regression on {len(df)} entries")
        num_columns = len(df.columns) - 1

        X = df.loc[:, df.columns != label_column].values
        y_pred = []

        for i in range(len(X)):
            o = 0
            for j in range(0, num_columns):
                o += X[i][j] * self.weights[j]
            y_pred += [o]

        return y_pred

    def fit_predict(self, df_trn, df_test, label_column):
        self.fit(df_trn, label_column)
        y_pred = self.predict(df_test, label_column)
        return y_pred
