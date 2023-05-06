import matplotlib.pyplot as plt
import numpy as np
import scipy

def display_alpha_digit(x):
    plt.imshow(x.reshape((20,16)), cmap='gray')
    plt.show()

def display_alpha_digits(X, save=False, title="image_generated"):
    plt.figure(figsize=(X.shape[0],10))
    for i in range(X.shape[0]):
        plt.subplot(1, X.shape[0], i+1)
        plt.imshow(X[i].reshape((20,16)),cmap='gray')
        plt.axis('off')
    if save:
        plt.savefig(f"{title}.png")
    plt.show()

def display_mnist_digit(x):
    plt.imshow(x.reshape((28,28)), cmap='gray')
    plt.show()

def display_mnist_digits(X):
    plt.figure(figsize=(X.shape[0],10))
    for i in range(X.shape[0]):
        plt.subplot(1, X.shape[0], i+1)
        plt.imshow(X[i].reshape((28,28)), cmap='gray')
        plt.axis('off')
    plt.show()

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
    