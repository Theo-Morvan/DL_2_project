from src.rbm import RBM

class DBN:
    def __init__(self, sizes):
        self.rbms = []
        for i in range(len(sizes) - 1):
            self.rbms.append(RBM(sizes[i], sizes[i + 1]))

    def train(self, X, lr, batch_size, nb_epochs):
        for rbm in self.rbms:
            rbm.train(X, lr, batch_size, nb_epochs)
            X = rbm.entree_sortie(X)

    def generer_image(self, nb_iter, nb_images):
        V = self.rbms[-1].generer_image(nb_iter, nb_images)
        for rbm in reversed(self.rbms[:-1]):
            V = rbm.sortie_entree(V)
        return V