# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy

import pandas

from feed_forward import FeedForwardNetwork
from linear_network import LinearClassifier
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

    print("\nHOUSE VOTES")

    # VIDEO PART 2
    # WEIGHT UPDATES FOR LOGISTIC REGRESSION
    print("LOGISTIC REGRESSION - WEIGHT UPDATES")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house = df_house.iloc[0:1]
    lr_reg = LinearClassifier(learn_rate=0.01, reg_value=0.5)
    lr_reg.fit(df_house, 'party', verbose=True)
    print("==============================\n")

    # VIDEO PART 1
    # PERFORMANCE FROM HOUSE VOTES ON ALL THREE NETWORKS
    # LOGISTIC
    print("LOGISTIC REGRESSION - RUN")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house_part = preprocessing.df_stratified_partition(df_house, 'party', 5)
    lr_reg = LinearClassifier(learn_rate=0.01, reg_value=0.5)
    scores = processing_linear.classify_cross_validation(df_house_part, lr_reg, 'party')
    print(f"Regularized Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # VIDEO PART 3
    # WEIGHT UPDATES FOR FEED FORWARD ON BOTH HIDDEN LAYERS
    print("FEED FORWARD - WEIGHT UPDATES")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house = preprocessing.encode_onehot(df_house, "party")
    df_house = df_house.iloc[0:1]
    ff = FeedForwardNetwork(learn_rate=0.1, num_hidden1=3, num_hidden2=3, type='Classifier')
    ff.fit(df_house, ['party_0', 'party_1'], verbose=True)
    print("==============================\n")

    # FEED FORWARD
    print("FEED FORWARD - RUN")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house_part = preprocessing.df_stratified_partition(df_house, 'party', 5)

    # after stratified partitioning we should one-hot encode the target so the NN knows how many output nodes to create
    for i in range(len(df_house_part)):
        df_house_part[i] = preprocessing.encode_onehot(df_house_part[i], "party")

    ff = FeedForwardNetwork(learn_rate=0.1, num_hidden1=3, num_hidden2=3, type='Classifier')
    scores = processing_feedforward.classify_cross_validation(df_house_part, ff, ['party_0', 'party_1'])
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # AUTOENCODE
    # print("AUTOENCODE")
    # df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    # df_house_part = preprocessing.df_stratified_partition(df_house, 'party', 5)
    #
    # # after stratified partitioning we should one-hot encode the target so the NN knows how many output nodes to create
    # for i in range(len(df_house_part)):
    #     df_house_part[i] = preprocessing.encode_onehot(df_house_part[i], "party")
    #
    # ae = AutoEncodedNetwork(encode_learn_rate=0.1, learn_rate=0.001, num_encoding=15, num_hidden1=6, type='Classifier')
    # scores = processing_feedforward.classify_cross_validation(df_house_part, ae, ['party_0', 'party_1'])
    # print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    # print("==============================\n")

    # VIDEO PART 4
    # WEIGHT UPDATES FOR AUTOENCODER

    # VIDEO PART 5
    # AUTOENCODER RECOVERY

    # VIDEO PART 6
    # GRADIENT CALCULATION FOR OUTPUT

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


