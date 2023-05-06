import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import pandas as pd
import numpy as np

def sampling(mu, log_var):
    # this function samples a Gaussian distribution,
    # with average (mu) and standard deviation specified (using log_var)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu) 

class VAE_Encoder(nn.Module):
    def __init__(self, input_size,middle_layer_size, latent_dim) -> None:
        super(VAE_Encoder,self).__init__()
        self.first_layer = nn.Linear(input_size, middle_layer_size)
        self.mu_net = nn.Linear(middle_layer_size, latent_dim)
        self.sigma_net = nn.Linear(middle_layer_size, latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.first_layer(x))
        return self.mu_net(x), self.sigma_net(x)
        

class VAE(nn.Module):

    def __init__(self, input_size, middle_layer_size, latent_dim ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size

        self.encoder = VAE_Encoder(input_size, middle_layer_size, latent_dim)
        self.mu_layer = nn.Linear(middle_layer_size, latent_dim)
        self.log_sig_layer = nn.Linear(middle_layer_size, latent_dim)


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, middle_layer_size),
            nn.ReLU(),
            nn.Linear(middle_layer_size, input_size),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
    
    def forward(self, x):
        z_mu, z_log_var = self.encoder(x)
        z = sampling(z_mu, z_log_var)
        return self.decoder(z), z_mu, z_log_var

    def objective_loss(self, x, x_reconstructed, mu, log_var):
        batch_size = x.shape[0]
        BCE = F.binary_cross_entropy(x_reconstructed.view(-1,784), x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_data:str,
        labels=None
    ):
        super(torch.utils.data.Dataset, self).__init__()
        data_train = pd.read_csv(path_data)
        if labels!= None:
            try:
                data_train = data_train[data_train['label'].isin(labels)]
            except:
                raise ValueError("the labels must be a list of integers between 0 to 9 inclusive")
        y_numerical =torch.Tensor(data_train['label'].values)
        try:
            self.y = F.one_hot(y_numerical.to(torch.int64)).to(torch.float32)
        except:
            self.y = y_numerical.to(torch.float32)
        X = torch.Tensor(data_train.drop('label', axis=1).values).to(torch.float32)
        self.X = (1*(X > 127)).to(torch.float32)


    def classes(self):
        """
        returns the labels of elements of the Dataset.
        """
        return self.y

    def __len__(self):
        return len(self.y)

    def get_batch_labels(self, idx):
        """
        Fetch a batch of labels
        args :
            - idx : set of indexes to query
        """
        return self.y[idx]

    def get_batch_data(self, idx):
        """
        Fetch a batch of inputs
        args :
            - idx : set of indexes to query
        """
        return self.X[idx]

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_data, batch_y

def train_model(model, learning_rate, train_loader,batch_size, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    for epoch in range(n_epochs):
        train_loss = 0
        for data, _ in train_loader:
            optimizer.zero_grad()
            data_reconstructed, z_mu, z_log_var = model(data)
            loss = model.objective_loss(data, data_reconstructed, z_mu, z_log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"[*] Epoch {epoch}, average loss : {train_loss/len(train_loader.dataset)}")

def generate_data(model, number_images):
    epsilon = torch.randn(number_images, 1, model.latent_dim)
    generations = model.decoder(epsilon)
    return generations
