import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import ipdb
from RBM_baseline import *
from DBN_baseline import *

class DNN(nn.Module):

    def __init__(
            self,
            hidden_layers_size:list,
            input_size:int,
            output_size:int
        ) -> None:
        super(DNN, self).__init__()

        self.input_size=input_size
        self.output_size = output_size
        self.layers_size = [input_size]+hidden_layers_size +[output_size]
        self.number_layers = len(self.layers_size)
        self.DBN = DBN(hidden_layers_size, input_size)
        self.w_class = torch.empty(
            size=(self.hidden_layers_size[-1],self.ouput_size)
        ).normal_(mean=0, std=0.01)
        self.b_class = torch.zeros(self.output_size)
    
    def calcul_softmax(self, X):
        return F.softmax(X @ self.w_class + self.b_class)

    def entree_sortie_reseau(self,X):
        pass
        