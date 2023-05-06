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

def lire_alpha_digit_full(caracteres:list):
    mat = scipy.io.loadmat('data/binaryalphadigs.mat')
    data = mat["dat"]
    NUM = 31
    if type(caracteres)==list:
        number_values = len(caracteres)
        X = np.zeros((number_values*39, 20*16))
        j = 0
        for caractere in caracteres:
            try:
                caractere = int(caractere)
                if caractere>9:
                    raise(ValueError('integer should be lower or equal than 9 '))
                position = caractere
            except:
                position = (ord(caractere)& NUM)+9
            for i in range(39):
                place = j*39 + i
                X[place,:] = data[position, i].flatten()
            j+=1
        return X          
    elif type(caracteres)==str:
        position = (ord(caracteres)& NUM)+9
        X = np.zeros((39, 20*16))
        for i in range(39):
            X[i,:] = data[position,i].flatten()
        return X
    elif type(caracteres)==int:
        if caracteres>9:
            raise(ValueError('integer should be lower or equal than 9 '))
        position = caracteres
        X = np.zeros((39, 20*16))
        for i in range(39):
            X[i,:] = data[position,i].flatten() 
        return X       
    else:
        raise(TypeError('input should be either a list, string or integer'))

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