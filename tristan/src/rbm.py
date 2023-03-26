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
    
    def train(self, X, lr, batch_size, nb_epochs, verbose=True):
        for epoch in range(nb_epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:min(i + batch_size, X.shape[0])].copy()
                t_b = X_batch.shape[0]
                v_0 = X[i:min(i + batch_size, X.shape[0])]
                p_h_v_0 = self.entree_sortie(v_0)
                h_0 = (np.random.rand(t_b,self.output_size)<p_h_v_0)*1
                p_v_h_0 = self.sortie_entree(h_0)
                v_1 = (np.random.rand(t_b,self.input_size)<p_v_h_0)*1
                h_1 = self.entree_sortie(v_1)
                p_h_v_1 = self.entree_sortie(v_1)

                grad_a = np.sum(v_0 - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)
                grad_W = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1

                self.a += lr / len(v_0) * grad_a
                self.b += lr / len(v_0) * grad_b
                self.W += lr / len(v_0) * grad_W
            if verbose:
                print("Epoch %d, erreur quadratique moyenne de reconstruction : %f" % (epoch, np.mean((X - self.sortie_entree(self.entree_sortie(X)))**2)))

    # def generer_image(self, nb_iter, nb_images):
    #     H = np.random.binomial(1, 0.5, (nb_images, self.output_size))
    #     for i in range(nb_iter):
    #         V = self.sortie_entree(H)
    #         H = self.entree_sortie(V)
    #     return V
    
    def generer_image(self, nb_iters_Gibbs, nb_images):
        images = np.zeros((nb_images, self.input_size))
        for i in range(nb_images):
            v = ((np.random.rand(self.input_size)<1/2)*1)     
            for iter_gibb in range(nb_iters_Gibbs):
                h = (np.random.rand(self.output_size)<self.entree_sortie(v))*1
                v = (np.random.rand(self.input_size)<self.sortie_entree(h))*1
            images[i,:] = v
        return images
