'''
NTHU EE Machine Learning HW2
Author: Jack Liu
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def GaussianBasis(x1, x2, s1, s2, u_i, u_j):
    return np.exp(-((x1 - u_i) ** 2 / (2 * s1 ** 2)) - ((x2 - u_j) ** 2 / (2 * s2 ** 2)))


def BuildPhi(data, x1_min, x2_min, x1_max, x2_max, s1, s2, O1, O2):
    phi_x = np.zeros(shape=(data.shape[0], O1 * O2 + 2))
    for idx in range(data.shape[0]):
        for i in range(1, O1 + 1):
            for j in range(1, O2 + 1):
                k = O2 * (i - 1) + j
                u_i = s1 * (i - 1) + x1_min
                u_j = s2 * (j - 1) + x2_min
                x1 = data[idx][0]
                x2 = data[idx][1]
                phi_x[idx][k - 1] = GaussianBasis(x1, x2, s1, s2, u_i, u_j)
        phi_x[idx][O1 * O2] = data[idx][2]
        phi_x[idx][O1 * O2 + 1] = 1
    return phi_x


def Select_O(data_train, data_test_feature, data_test_label):
    # find O1, O2 based on BLR
    O1_BLR = 0
    O2_BLR = 0
    O1_list = []
    O2_list = []
    mse_list = []
    blr_mse = float('inf')
    for i in range(2, 10):
        for j in range(2, 10):
            y_pred = BLR(data_train, data_test_feature, i, j)
            mse = CalMSE(y_pred, data_test_label)
            O1_list.append(i)
            O2_list.append(j)
            mse_list.append(mse)
            if mse < blr_mse:
                blr_mse = mse
                O1_BLR = i
                O2_BLR = j
    print(f'Best BLR MSE: {blr_mse}, Best O1: {O1_BLR}, Best O2: {O2_BLR}')

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(O1_list, O2_list, mse_list, alpha=0.8, label='BLR', edgecolor='black')
    ax.set_xlabel('O1')
    ax.set_ylabel('O2')
    ax.set_zlabel('MSE loss')
    plt.legend(loc='upper left')
    plt.suptitle('O1, O2 vs. MSE loss(BLR)')
    plt.show()

    # find O1, O2 based on MLR
    O1_MLR = 0
    O2_MLR = 0
    mlr_mse = float('inf')
    O1_list = []
    O2_list = []
    mse_list = []
    for i in range(2, 10):
        for j in range(2, 10):
            y_pred = MLR(data_train, data_test_feature, i, j)
            mse = CalMSE(y_pred, data_test_label)
            O1_list.append(i)
            O2_list.append(j)
            mse_list.append(mse)
            if mse < mlr_mse:
                mlr_mse = mse
                O1_MLR = i
                O2_MLR = j
    print(f'Best MLR MSE: {mlr_mse}, Best O1: {O1_MLR}, Best O2: {O2_MLR}')

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(O1_list, O2_list, mse_list, alpha=0.8, label='MLR', edgecolor='black')
    ax.set_xlabel('O1')
    ax.set_ylabel('O2')
    ax.set_zlabel('MSE loss')
    plt.legend(loc='upper left')
    plt.suptitle('O1, O2 vs. MSE loss(MLR)')
    plt.show()


# do not change the name of this function
def BLR(train_data, test_data_feature, O1=2, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    # find train feature and label
    train_data_feature = train_data[:, :3]
    t = train_data[:, 3]

    # find x1, x2 (min and max) and s1, s2
    x1_min, x1_max = min(train_data_feature[:, 0]), max(train_data_feature[:, 0])
    x2_min, x2_max = min(train_data_feature[:, 1]), max(train_data_feature[:, 1])
    s1 = (x1_max - x1_min) / (O1 - 1)
    s2 = (x2_max - x2_min) / (O2 - 1)

    # Use phi_x for training
    phi_train_x = BuildPhi(train_data_feature, x1_min, x2_min, x1_max, x2_max, s1, s2, O1, O2)

    # Solve the w
    # lambda = alpha / beta => It means the regularization term
    alpha = 1.0
    beta = 1.0
    Sn = np.linalg.inv(alpha * np.identity(phi_train_x.shape[1]) + beta * phi_train_x.T.dot(phi_train_x))
    w = beta * Sn.dot(phi_train_x.T).dot(t)
    
    # Use phi_x for testing
    phi_test_x = BuildPhi(test_data_feature, x1_min, x2_min, x1_max, x2_max, s1, s2, O1, O2)

    # prediction
    y_BLRprediction = np.array([np.sum(w * phi_test_x[p]) for p in range(phi_test_x.shape[0])])
    return y_BLRprediction


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    # find train feature and label
    train_data_feature = train_data[:, :3]
    t = train_data[:, 3]
    
    # find x1, x2 (min and max) and s1, s2
    x1_min, x1_max = min(train_data_feature[:, 0]), max(train_data_feature[:, 0])
    x2_min, x2_max = min(train_data_feature[:, 1]), max(train_data_feature[:, 1])
    s1 = (x1_max - x1_min) / (O1 - 1)
    s2 = (x2_max - x2_min) / (O2 - 1)

    # Use phi_x for training
    phi_train_x = BuildPhi(train_data_feature, x1_min, x2_min, x1_max, x2_max, s1, s2, O1, O2)
    
    # Solve the w
    w = np.linalg.inv(phi_train_x.T.dot(phi_train_x)).dot(phi_train_x.T).dot(t)
    
    # Use phi_x for testing
    phi_test_x = BuildPhi(test_data_feature, x1_min, x2_min, x1_max, x2_max, s1, s2, O1, O2)

    # prediction
    y_MLLSprediction = np.array([np.sum(w * phi_test_x[p]) for p in range(phi_test_x.shape[0])])
    return y_MLLSprediction


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=2)
    parser.add_argument('-O2', '--O_2', type=int, default=2)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    # Select_O(data_train, data_test_feature, data_test_label)
    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()