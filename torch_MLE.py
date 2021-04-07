import torch
import os
from scipy.io import mmread
import torch.optim as optim
import torch.nn as nn
from Adjacency_matrix import Preprocessing
from torch_sparse import spspmm

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
    U, V, values = [], [], []
    for i in range(lenght):
        #data = data[i].split(" ")
        U.append(int(data[i].split(" ")[0]))
        V.append(int(data[i].split(" ")[1]))
        values.append(int(data[i].split(" ")[2]))
    U, V, values = torch.tensor(U), torch.tensor(V), torch.tensor(values)
    return U, V, values

U, V, values = loader(text_file)

#Loading data and making adjancency matrix
raw_data = mmread(text_file)
#print(raw_data)

A = raw_data.todense()
A = torch.tensor(A)
#print(A.shape)

class LSM(nn.Module):
    def __init__(self, B, input_size, latent_dim, sample_size):
        super(LSM, self).__init__()
        self.A = B
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1]))

        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.latent_dim))

        # Sampling
        self.sample.size = sample_size


    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm
        # sample for undirected network
        sample_idx = torch.multinomial(self.sampling_weights, self.sample_size, replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator = torch.cat([sample_idx.unsqueeze(0), sample_idx.unsqueeze(0)], 0)
        # adjacency matrix in edges format
        edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges, torch.ones(edges.shape[1]), indices_translator,
                                torch.ones(indices_translator.shape[1]), self.input_size, self.input_size,
                                self.input_size, coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_translator, torch.ones(indices_translator.shape[1]), indexC, valueC,
                                self.input_size, self.input_size, self.input_size, coalesced=True)

        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_idx, sparse_i_sample, sparse_j_sample

    def log_likelihood(self, A):

        z_dist = (((torch.unsqueeze(self.latent_zi, 1) - self.latent_zj+1e-06)**2).sum(-1))**0.5
        bias_matrix = torch.unsqueeze(self.beta, 1) + self.gamma
        Lambda = bias_matrix - z_dist
        LL = (A*Lambda).sum() - torch.sum(torch.exp(Lambda))

        return LL
''' 
    def optimizer(self, iterations):
        # Implements stochastic gradient descent (optionally with momentum). Nesterov momentum
        for _ in range(iterations):
            optimizer = optim.SGD(params=self.parameters(), lr=0.01, momentum=0.9)
            loss = -self.log_likelihood(self.A)/ self.input_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss
'''

if __name__ == "__main__":
    preproc = Preprocessing()
#    model = LSM(B=preproc.From_Biadjacency_To_Adjacency(A), input_size=A.shape[0], latent_dim=2)
    model = LSM(B=A, input_size=A.shape, latent_dim=2)
#    B = model.optimizer(10)
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    for _ in range(10):
        loss = -model.log_likelihood(model.A) / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())






