import numpy as np
from tqdm import tqdm

from src.dbn import DBN
from scipy.special import softmax

class DNN:
    def __init__(self, sizes):
        self.dbn = DBN(sizes[:-1])
        self.W = np.random.normal(0, 0.01, (sizes[-2], sizes[-1]))
        self.b = np.zeros(sizes[-1])

    def pretrain(self, X, lr, batch_size, nb_epochs, verbose=True):
        self.dbn.train(X, lr, batch_size, nb_epochs, verbose=verbose)
    
    def calcul_softmax(self, X):
        return softmax(np.dot(X, self.W) + self.b, axis=1)
    
    def entree_sortie(self, X):
        sorties = [X]
        for rbm in self.dbn.rbms:
            X = rbm.entree_sortie(X)
            sorties.append(X)
        sorties.append(self.calcul_softmax(X))
        return sorties
    
    def retropropagation(self, X, y, lr, batch_size, nb_epochs, verbose=True):
        iterator = tqdm(range(nb_epochs)) if not verbose else range(nb_epochs)
        if not verbose:
            iterator.set_description(f"Retropropagation | Cross entropy loss: {None} | Epoch")
        
        losses = []
        for epoch in iterator:
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:min(i+batch_size, X.shape[0])]
                y_batch = y[i:min(i+batch_size, y.shape[0])]
                
                n = X_batch.shape[0]
                sorties = self.entree_sortie(X_batch)

                dZ = [None for i in range(len(sorties))]
                dA = [None for i in range(len(sorties))]
                dW = [None for i in range(len(sorties))]
                db = [None for i in range(len(sorties))]

                dZ[-1] = sorties[-1] - y_batch
                dW[-1] = np.dot(sorties[-2].T, dZ[-1]) / n
                db[-1] = np.sum(dZ[-1], axis=0) / n
                dA[-2] = np.dot(dZ[-1], self.W.T)

                for i in range(len(sorties)-2, 0, -1):
                    dZ[i] = dA[i] * sorties[i] * (1 - sorties[i])
                    dW[i] = np.dot(sorties[i-1].T, dZ[i]) / n
                    db[i] = np.sum(dZ[i], axis=0) / n
                    dA[i-1] = np.dot(dZ[i], self.dbn.rbms[i-1].W.T)

                self.W -= lr * dW[-1]
                self.b -= lr * db[-1]
                for i in range(len(sorties)-2, 0, -1):
                    self.dbn.rbms[i - 1].W -= lr * dW[i]
                    self.dbn.rbms[i - 1].b -= lr * db[i]

            sorties = self.entree_sortie(X)
            cross_entropy = -np.sum(y * np.log(sorties[-1])) / X.shape[0]
            losses.append(cross_entropy)
            if verbose:
                print(f"Epoch {epoch+1}/{nb_epochs} - Cross entropy: {cross_entropy}")
            else:
                iterator.set_description(f"Retropropagation | Cross entropy loss: {cross_entropy:.4f} | Epoch")
        return losses

    def test(self, X, y):
        sorties = self.entree_sortie(X)
        y_pred = np.argmax(sorties[-1], axis=1)
        y_true = np.argmax(y, axis=1)
        return np.sum(y_pred != y_true) / y.shape[0]