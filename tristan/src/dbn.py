from tqdm import tqdm
from src.rbm import RBM
import numpy as np

class DBN:
    def __init__(self, sizes):
        self.rbms = []
        for i in range(len(sizes) - 1):
            self.rbms.append(RBM(sizes[i], sizes[i + 1]))

    def train(self, X, lr, batch_size, nb_epochs, verbose=True):
        iterator = tqdm(self.rbms) if not verbose else self.rbms
        if not verbose:
            iterator.set_description(f"DBN Training | RBM")
        for rbm in iterator:
            rbm.train(X, lr, batch_size, nb_epochs, verbose=verbose)
            X = rbm.entree_sortie(X)

    # def generer_image(self, nb_iter, nb_images):
    #     V = self.rbms[-1].generer_image(nb_iter, nb_images)
    #     for rbm in reversed(self.rbms[:-1]):
    #         V = rbm.sortie_entree(V)
    #     return V
    
    def generer_image(self, nb_iters_Gibbs, nb_images):
        for j in range(nb_images):
            v = self.rbms[-1].generer_image(nb_iters_Gibbs, nb_images)
            for i, rbm in enumerate(reversed(self.rbms[:-1])):
                v = ((np.random.rand(rbm.input_size)<rbm.sortie_entree(v))*1)
        return v