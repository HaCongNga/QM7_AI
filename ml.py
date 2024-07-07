import numpy as np
import scipy
from scipy import io

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import argparse

def load_dataset() :
    dataset = scipy.io.loadmat('qm7.mat')
    return dataset

def get_train_test_index(dataset, split) :
    """
    'split' is an integer ranging from 0 to 4.
    For cross validation, train is the subset excluding the 'split'; test is the 'split' subset
    Return train_set and test_set indices
    """
    splits = 5
    dataset = dataset['P']
    train_set = dataset[list(range(0, split)) + list(range(split+1, splits))].flatten()
    test_set = dataset[split]

    return train_set, test_set

def get_features(dataset, indices) :
    """
    Input : 
        + index is the set of indices of features
    Output :
        Return the corresponding features and labels for training and testing ML models
    """

    # Get coulomb matrices X
    X = dataset['X']
    X = X.reshape(X.shape[0], -1)
    X = X[indices]
    print("Shape of feature X : ", X.shape)

    # Get atomic charge Z :
    Z = dataset['Z']
    Z = Z[indices]
    print("Shape of feature atomic charge Z : ", Z.shape)

    # Get the Cartesian coordinate :
    R = dataset['R']
    R = R[indices]
    R = R.reshape(R.shape[0], -1)
    print("Shape of cartesian coordinate R : ", R.shape)

    # Concate all features :
    features = np.concatenate((X,Z), axis=1)
    print("Shape of features : ", features.shape)

    T = dataset['T']
    T = np.squeeze(T)
    print("Shape T : ", T.shape)
    labels = T[indices]
    print("Shape of labels : ", labels.shape)

    return features, labels


def get_model(model_name) :
    if model_name == "LR" :
        return LinearRegression()
    elif model_name == "SVR" :
        return make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    elif model_name == "Gauss" :
        kernel = DotProduct() + WhiteKernel()
        return GaussianProcessRegressor(kernel=kernel, alpha=10, random_state=0)
    else :
        raise ValueError("Unsupported model !")

def main() :
    dataset = load_dataset()

    parser = argparse.ArgumentParser(description="Process some arguments")
    parser.add_argument('model_name', type=str, default="LR", help="SVR")
    args = parser.parse_args()

    # Get ml model 
    model = get_model(args.model_name)

    splits = 5
    MAEs = []
    RMSEs = []
    train_scores = []

    for i in range(splits) :
        train_indices, test_indices = get_train_test_index(dataset, i)
        train_features, train_labels = get_features(dataset, train_indices)
        test_features, test_labels = get_features(dataset, test_indices)

        reg = model.fit(train_features, train_labels)
        score = reg.score(train_features, train_labels)
        train_scores.append(score)
        prediction = reg.predict(test_features)
        mae = np.abs(prediction-test_labels).mean(axis=0)
        print("mae : ", mae)
        MAEs.append(mae)
        rmse = np.square(prediction-test_labels).mean(axis=0)**.5
        print("rmse : ", rmse)
        RMSEs.append(rmse)
        print(f"iteration {i}, train score : {score}, MAE : {mae}, RMSE : {rmse}")
    print("Mean MAEs, RMSEs :", sum(MAEs)/len(MAEs), sum(RMSEs)/len(RMSEs))

if __name__ == '__main__' :
    main()

