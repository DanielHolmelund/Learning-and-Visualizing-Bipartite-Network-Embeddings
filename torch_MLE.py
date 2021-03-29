import torch
import os
from scipy.io import mmread
import torch.optim as optim
import torch.nn as nn
from Adjacency_matrix import Preprocessing

#Creating dataset

os.chdir('Datasets/Single_cell')

text_file = 'Test.mtx'

def loader(text_file):
    """
    :param text_file:
    :return:
    """
    f = open(text_file, "r")
    data = f.read()
    data = data.split("\n")
    lenght = len(data)
    U, V, values = torch.tensor([lenght]), torch.tensor([lenght]), torch.tensor([lenght])
    for i in range(len(data)):
        data = data[i].split(" ")
        U[i] = int(data[0])
        V[i] = int(data[1])
        values[i] = int(data[2])
    return U, V, values

with open(text_file, "r") as f:
    loaded_list = f.read().split("\n")
loaded_tensor = torch.tensor(loaded_list)

#Loading data and making adjancency matrix
raw_data = mmread(text_file)
#print(raw_data)

A = torch.sparse_coo_tensor(raw_data)
print(A)
A = torch.tensor(A)
print(A)
x = 1
#print(A.shape)

class LSM(nn.Module):
    def __init__(self, A, input_size, latent_dim):
        super(LSM, self).__init__()
        self.A = A
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1]))

        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.latent_dim))

    def log_likelihood(self, A):

        z_dist = (((torch.unsqueeze(self.latent_zi, 1) - self.latent_zj+1e-06)**2).sum(-1))**0.5
        bias_matrix = torch.unsqueeze(self.beta, 1) + self.gamma
        Lambda = bias_matrix - z_dist
        LL = (A*Lambda).sum() - torch.sum(torch.exp(Lambda))

        return LL

if __name__ == "__main__":
    model = LSM(A=A, input_size=A.shape, latent_dim=2)
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    for _ in range(10):
        loss = -model.log_likelihood(model.A) / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())






