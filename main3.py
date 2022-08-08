import random

import pandas

from autoencoder import AutoEncodedNetwork
import dataprep
import eval
import numpy as np
import pandas as pd
import preprocessing
import processing_feedforward

# Press the green button in the gutter to run the script.
import processing_linear

if __name__ == '__main__':

    # ABALONE
    print("ABALONE")
    df_abalone = dataprep.prepare_abalone('datasets/abalone.data')
    df_abalone_partitions = preprocessing.df_partition(df_abalone, 5)
    ae = AutoEncodedNetwork(encode_learn_rate=0.1, learn_rate=0.01, num_encoding=9, num_hidden1=6, type='Regressor')
    scores = processing_feedforward.regression_cross_validation(df_abalone_partitions, ae, ['Rings'])
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")
    print("==============================\n")

    # BREAST CANCER
    print("BREAST CANCER")
    df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_breast_part = preprocessing.df_stratified_partition(df_breast, 'Class', 5)

    # after stratified partitioning we should one-hot encode the target so the NN knows how many output nodes to create
    for i in range(len(df_breast_part)):
        df_breast_part[i] = preprocessing.encode_onehot(df_breast_part[i], "Class")

    # now feed it into the classifier
    ae = AutoEncodedNetwork(encode_learn_rate=0.1, learn_rate=0.01, num_encoding=8, num_hidden1=6, type='Classifier')
    scores = processing_feedforward.classify_cross_validation(df_breast_part, ae, ['Class_0', 'Class_1'])
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # CAR
    print("CAR")
    df_car = dataprep.prepare_car('datasets/car.data')
    df_car_part = preprocessing.df_stratified_partition(df_car, 'CAR', 5)

    # after stratified partitioning we should one-hot encode the target so the NN knows how many output nodes to create
    for i in range(len(df_car_part)):
        df_car_part[i] = preprocessing.encode_onehot(df_car_part[i], "CAR")

    # now feed it into the classifier
    ae = AutoEncodedNetwork(encode_learn_rate=0.1, learn_rate=0.01, num_encoding=4, num_hidden1=6, type='Classifier')
    scores = processing_feedforward.classify_cross_validation(df_car_part, ae, ['CAR_unacc', 'CAR_acc', 'CAR_good', 'CAR_vgood'])
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # FOREST FIRES
    print("FOREST FIRES")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_part = preprocessing.df_partition(df_forest, 5)
    ae = AutoEncodedNetwork(encode_learn_rate=0.01, learn_rate=0.01, num_encoding=11, num_hidden1=6, type='Regressor')
    scores = processing_feedforward.regression_cross_validation(df_forest_part, ae, ['area'])
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")
    print("==============================\n")

    # HOUSE VOTES
    print("HOUSE VOTES")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house_part = preprocessing.df_stratified_partition(df_house, 'party', 5)

    # after stratified partitioning we should one-hot encode the target so the NN knows how many output nodes to create
    for i in range(len(df_house_part)):
        df_house_part[i] = preprocessing.encode_onehot(df_house_part[i], "party")

    ae = AutoEncodedNetwork(encode_learn_rate=0.1, learn_rate=0.001, num_encoding=15, num_hidden1=6, type='Classifier')
    scores = processing_feedforward.classify_cross_validation(df_house_part, ae, ['party_0', 'party_1'])
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # MACHINE
    print("MACHINE")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_partitions = preprocessing.df_partition(df_machine, 5)
    ae = AutoEncodedNetwork(encode_learn_rate=0.1, learn_rate=0.01, num_encoding=8, num_hidden1=6, type='Regressor')
    scores = processing_feedforward.regression_cross_validation(df_machine_partitions, ae, ['PRP'])
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")
    print("==============================\n")