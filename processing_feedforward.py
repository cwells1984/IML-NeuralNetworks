import copy
import eval
import numpy as np
import pandas as pd


# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def classify_cross_validation(df_trn_partitions, model, label_columns):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # Test the model and record its score
        y_pred, y_truth = model.fit_predict(df_trn_fold, df_test, label_columns)

        # Record the softmax values of y_pred and y_truth
        y_pred_sm = []
        for y_pred_value in y_pred:
            y_pred_sm += [np.argmax(eval.softmax(y_pred_value))]
        y_truth_sm = []
        for y_truth_value in y_truth:
            y_truth_sm += [np.argmax(y_truth_value)]

        # Print details about this fold
        score = eval.eval_softmax(y_truth, y_pred)
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), Acc= {score*100:.2f}%')
        if i == 0:
            print(f"Expected value of item 0 = {y_truth_sm[0]}, Actual Value = {y_pred_sm[0]}")
        scores += [score]

    return scores


# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def regression_cross_validation(df_trn_partitions, model, label_columns):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # Fit and make predictions
        y_pred, y_truth = model.fit_predict(df_trn_fold, df_test, label_columns)

        # Print details about this fold
        score = eval.eval_mse(y_truth, y_pred)[0]
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), MSE= {score:.2f}')
        if i == 0:
            print(f"Expected value of item 0 = {y_truth[0][0]:.2f}, Actual Value = {y_pred[0][0]:.2f}")
        scores += [score]

    return scores
