import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import ipdb
from RBM_baseline import *

class DBN(nn.Module):

    def __init__(
        self,
        hidden_layers_size:list,
        input_size:int,
    ) -> None:
        super(DBN,self).__init__()

        self.hidden_layers_size = [input_size]+hidden_layers_size
        self.number_layers = len(hidden_layers_size)
        DBN_RBMs = []
        for i in range(self.number_layers):
            DBN_RBMs.append(RBM(p=self.hidden_layers_size[i],q=self.hidden_layers_size[i+1]))
        self.DBN_RBMs = DBN_RBMs
        self.input_size = input_size
    
    def train_DBN(
        self,
        X:torch.tensor, 
        lr:float, 
        batch_size:int, 
        nb_epochs:int
    ):
        for i in range(self.number_layers):
            self.DBN_RBMs[i].train_RBM(X, lr, batch_size, nb_epochs)
            X = self.DBN_RBMs[i].entree_sortie_RBM(X)

    # def generate_image_DBN(self, nb_iters_Gibbs, nb_images):
    #     images = torch.zeros((self.input_size, nb_images))
    #     for i in range(nb_images):
    #         v = ((torch.rand(self.p)<1/2)*1).type(torch.float)
    #         h_forward = v     
    #         for iter_gibb in range(nb_iters_Gibbs):
    #             for j in range(self.number_layers):
    #                 h_forward = ((torch.rand(self.hidden_layers_size[j+1]) \
    #                               <self.DBN_RBMs[j].entree_sortie_RBM(h_forward))*1).type(torch.float)
    #             v_backward = h_forward
    #             for j in range(self.number_layers):
    #                 v_backward = ((torch.rand(self.hidden_layers_size[-(j+1)]) \
    #                               <self.DBN_RBMs[-(j+2)].sortie_entree_RBM(v_backward))*1).type(torch.float)
    #             h_forward = v_backward
            
    #         images[i, :] = v_backward
    #     return images
    
    def generate_image_DBN(self, nb_iters_Gibbs, nb_images):
        for j in range(nb_images):
            try:
                v = self.DBN_RBMs[-1].generer_image_RBM(nb_iters_Gibbs, nb_images)
                for i in range(2,self.number_layers+1):

                    v = ((torch.rand(self.hidden_layers_size[-(i+1)])<self.DBN_RBMs[-i].sortie_entree_RBM(v))*1).type(torch.float)
            except:
                ipdb.set_trace()
            
        return v.detach().cpu().numpy()