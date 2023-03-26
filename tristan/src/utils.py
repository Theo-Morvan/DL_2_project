import matplotlib.pyplot as plt

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