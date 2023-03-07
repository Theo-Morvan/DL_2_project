import scipy
import numpy as np
import pandas as pd

def get_indices(char):
    if char.isdigit():
        return int(char)
    else:
        return ord(char) - ord('A') + 10

def lire_alpha_digits(path, char):
    mat = scipy.io.loadmat(path)
    data = mat["dat"]
    start = get_indices(char)
    X = np.zeros((39, 20*16))
    for i in range(39):
        X[i,:] = data[start, i].flatten()
    return X

def lire_mnist(path_train, path_test):
    mnist_train = pd.read_csv(path_train)
    mnist_test = pd.read_csv(path_test)

    y_train_numerical = mnist_train['label'].to_numpy()
    X_train = mnist_train.drop('label', axis=1).to_numpy()
    y_test_numerical = mnist_test['label'].to_numpy()
    X_test = mnist_test.drop('label', axis=1).to_numpy()

    # transform y to one-hot encoding
    y_train = np.eye(10)[y_train_numerical]
    y_test = np.eye(10)[y_test_numerical]

    # convert gray scale (0-255) to binary (0-1) with 1 if pixel is > 127
    X_train = (X_train > 127).astype(int)
    X_test = (X_test > 127).astype(int)
    
    return X_train, y_train, X_test, y_test