# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
from linear_network import LinearMultiClassifier
from linear_network import LinearClassifier
from linear_network import LinearRegressor
import dataprep
import eval
import numpy as np
import pandas as pd
import preprocessing


# Press the green button in the gutter to run the script.
import processing_linear

if __name__ == '__main__':

    # LINEAR NETWORKS

    # ABALONE
    print("ABALONE")
    df_abalone = dataprep.prepare_abalone('datasets/abalone.data')
    df_abalone_part = preprocessing.df_partition(df_abalone, 5)
    lr = LinearRegressor(learn_rate=0.01, reg_value=0.5)
    scores = processing_linear.regression_cross_validation(df_abalone_part, lr, 'Rings')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")
    print("==============================\n")

    # BREAST CANCER
    print("BREAST CANCER")
    df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_breast_part = preprocessing.df_stratified_partition(df_breast, 'Class', 5)
    lr_reg = LinearClassifier(learn_rate=0.01, reg_value=0.5)
    scores = processing_linear.classify_cross_validation(df_breast_part, lr_reg, 'Class')
    print(f"Regularized Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # CAR
    print("CAR")
    df_car = dataprep.prepare_car('datasets/car.data')
    df_car_part = preprocessing.df_stratified_partition(df_car, 'CAR', 5)
    lr = LinearMultiClassifier(learn_rate=0.01, reg_value=0.5)
    scores = processing_linear.classify_cross_validation(df_car_part, lr, 'CAR')
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # FOREST FIRES
    print("FOREST FIRES")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_part = preprocessing.df_partition(df_forest, 5)
    lr = LinearRegressor(learn_rate=0.01, reg_value=0.5)
    scores = processing_linear.regression_cross_validation(df_forest_part, lr, 'area')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")
    print("==============================\n")

    # HOUSE VOTES
    print("HOUSE VOTES")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house_part = preprocessing.df_stratified_partition(df_house, 'party', 5)
    lr_reg = LinearClassifier(learn_rate=0.01, reg_value=0.5)
    scores = processing_linear.classify_cross_validation(df_house_part, lr_reg, 'party')
    print(f"Regularized Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    print("==============================\n")

    # MACHINE
    print("MACHINE")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_partitions = preprocessing.df_partition(df_machine, 5)
    lr = LinearRegressor(learn_rate=0.01, reg_value=0.5)
    scores = processing_linear.regression_cross_validation(df_machine_partitions, lr, 'PRP')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")
    print("==============================\n")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/


