import numpy as np

from scipy.special import expit as sigmoid

class RBM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.a = np.zeros(input_size)
        self.b = np.zeros(output_size)
        self.W = np.random.normal(0, 0.01, (input_size, output_size))

    def entree_sortie(self, V):
        return sigmoid(V @ self.W + self.b)
    
    def sortie_entree(self, H):
        return sigmoid(H @ self.W.T + self.a)
    
    def train(self, X, lr, batch_size, nb_epochs):
        for epoch in range(nb_epochs):
            for i in range(0, len(X), batch_size):
                v_0 = X[i:min(i + batch_size, X.shape[0])]
                h_0 = self.entree_sortie(v_0)
                v_1 = self.sortie_entree(h_0)
                h_1 = self.entree_sortie(v_1)
                
                grad_a = np.mean(v_0 - v_1, axis=0)
                grad_b = np.mean(h_0 - h_1, axis=0)
                grad_W = v_0.T @ h_0 - v_1.T @ h_1

                self.a += lr / len(v_0) * grad_a
                self.b += lr / len(v_0) * grad_b
                self.W += lr / len(v_0) * grad_W
            print("Epoch %d, erreur quadratique moyenne de reconstruction : %f" % (epoch, np.mean((X - self.sortie_entree(self.entree_sortie(X)))**2)))

    def generer_image(self, nb_iter, nb_images):
        H = np.random.binomial(1, 0.5, (nb_images, self.output_size))
        for i in range(nb_iter):
            V = self.sortie_entree(H)
            H = self.entree_sortie(V)
        return V
