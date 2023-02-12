import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import ipdb
from RBM_baseline import *
from DBN_baseline import *
import matplotlib.pyplot as plt

if __name__=="__main__":
    n = 200
    q = 40
    X = torch.tensor(lire_alpha_digit(["a"])).type(torch.float)
    p = X.shape[1]
    rbm= RBM(p, q)
    try:
        V = rbm.entree_sortie_RBM(X)
        X_rec = rbm.sortie_entree_RBM(V)
        print("hello")
        
    except:
        ipdb.set_trace()
    rbm.train_RBM(X, 0.01, 32, 20)
    image = rbm.generer_image_RBM(2,1)
    
    
    image = image.detach().numpy().reshape((20,16))
    dbn = DBN([50,50,50], 320)
    dbn.train_DBN(X, 0.01, 16, 2000)

    images = dbn.generate_image_DBN(2,2)
    image = images[0]
    plt.imshow(image.reshape((20,16)),interpolation='nearest')
    plt.show()
