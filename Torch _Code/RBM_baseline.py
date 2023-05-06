import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import ipdb


class RBM(nn.Module):

    def __init__(
        self,
        p: int,
        q: int, 
    ) -> None:
        super(RBM,self).__init__()
        
        self.p = p
        self.q = q
        self.a = torch.zeros(p)
        self.b = torch.zeros(q)
        self.w = torch.empty(size=(self.p,self.q)).normal_(mean=0, std=0.01)
    
    def entree_sortie_RBM(self, V):
        return torch.sigmoid(self.b + V @ self.w)
    
    def sortie_entree_RBM(self,H):
        return torch.sigmoid(self.a + H @ self.w.T)
    
    def train_RBM(self, X, lr, batch_size, nb_epochs):
        for epoch in range(nb_epochs):
            X_copy = X.clone()

            X_copy = X_copy[torch.randperm(X_copy.shape[0]),:]
            number_batch = X.shape[0] // batch_size + 1
            for batch in range(number_batch):
                X_batch = X_copy[
                    batch*batch_size:min(batch*batch_size+batch_size, X_copy.shape[0]),
                    :
                ]
                t_b = X_batch.shape[0]
                # print(t_b)
                # ipdb.set_trace()
                v_0 = X_batch
                p_h_v_0 = self.entree_sortie_RBM(v_0)
                h_0 = ((torch.rand(size=(t_b,self.q))<p_h_v_0)*1).type(torch.float)
                try:
                    p_v_h_0 = self.sortie_entree_RBM(h_0)
                except:
                    ipdb.set_trace()
                v_1 = ((torch.rand(size=(t_b,self.p))<p_v_h_0)*1).type(torch.float)
                p_h_v_1 = self.entree_sortie_RBM(v_1)
                grad_a = torch.sum(v_0-v_1,dim=0)
                grad_b = torch.sum(p_h_v_0 - p_h_v_1, dim=0)
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1
                self.w = self.w + (lr/t_b)*grad_w
                try:
                    self.a = self.a + (lr/t_b)*grad_a
                except:
                    ipdb.set_trace()
                self.b = self.b + (lr/t_b)*grad_b
            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            if (epoch+1)%50==0:
                print(f" reconstruction error at epoch {epoch+1} :{((X-X_rec)**2).mean()}")
    
    def generer_image_RBM(self, nb_iters_Gibbs, nb_images):
        images = torch.zeros((nb_images, self.p))
        for i in range(nb_images):
            v = ((torch.rand(self.p)<1/2)*1).type(torch.float)     
            for iter_gibb in range(nb_iters_Gibbs):
                h = ((torch.rand(self.q)<self.entree_sortie_RBM(v))*1).type(torch.float)
                v = ((torch.rand(self.p)<self.sortie_entree_RBM(h))*1).type(torch.float)
            images[i,:] = v
        return images

def lire_alpha_digit(caracteres:list):
    mat = scipy.io.loadmat('binaryalphadigs.mat')
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
    