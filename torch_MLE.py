import torch
import os
from scipy.io import mmread
import torch.optim as optim
import torch.nn as nn
from Adjacency_matrix import Preprocessing
from torch_sparse import spspmm
import pandas as pd
import numpy as np
#Creating dataset
from blobs import *

#os.chdir('Datasets/divorce/')

#text_file = 'divorce.mtx'


#Loading data and making adjancency matrix
#raw_data = mmread(text_file)



#A = raw_data.todense()

#A = torch.tensor(A)



class LSM(nn.Module):
    def __init__(self, A, input_size, latent_dim, sparse_i_idx, sparse_j_idx, count, sample_i_size, sample_j_size):
        super(LSM, self).__init__()
        self.A = A
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1]))

        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.latent_dim))
        #Change sample weights for each partition
        self.sampling_i_weights = torch.ones(input_size[0])
        self.sampling_j_weights = torch.ones(input_size[1])
        #Change sample sizes for each partition
        self.sample_i_size = sample_i_size
        self.sample_j_size = sample_j_size

        self.sparse_i_idx = sparse_i_idx
        self.sparse_j_idx = sparse_j_idx

        self.count = count



#        self.myparameters = nn.ParameterList([self.latent_zi, self.latent_zj, self.beta, self.gamma])
        #self.nn_layers = nn.Modulelist([self.latent_zi, self.latent_zj])

    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for bipartite network
        sample_i_idx = torch.multinomial(self.sampling_i_weights, self.sample_i_size, replacement=False)
        sample_j_idx = torch.multinomial(self.sampling_j_weights, self.sample_j_size, replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_i_translator = torch.cat([sample_i_idx.unsqueeze(0), sample_i_idx.unsqueeze(0)], 0)
        indices_j_translator = torch.cat([sample_j_idx.unsqueeze(0), sample_j_idx.unsqueeze(0)], 0)
        # adjacency matrix in edges format
        edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges, self.count.float(), indices_j_translator,
                                torch.ones(indices_j_translator.shape[1]), self.input_size[0], self.input_size[1],
                                self.input_size[1], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_i_translator, torch.ones(indices_i_translator.shape[1]), indexC, valueC,
                                self.input_size[0], self.input_size[0], self.input_size[1], coalesced=True)

        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_i_idx, sample_j_idx, sparse_i_sample, sparse_j_sample, valueC

    def log_likelihood(self):
        sample_i_idx, sample_j_idx, sparse_i_sample, sparse_j_sample, valueC = self.sample_network()
        z_dist = (((torch.unsqueeze(self.latent_zi[sample_i_idx], 1) - self.latent_zj[sample_j_idx]+1e-06)**2).sum(-1))**0.5
        bias_matrix = torch.unsqueeze(self.beta[sample_i_idx], 1) + self.gamma[sample_j_idx]
        Lambda = bias_matrix - z_dist
        z_dist_links = (((self.latent_zi[sparse_i_sample] - self.latent_zj[sparse_j_sample]+1e-06)**2).sum(-1))**0.5
        bias_links = self.beta[sparse_i_sample] + self.gamma[sparse_j_sample]
        log_Lambda_links = (bias_links - z_dist_links) * valueC
        LL = log_Lambda_links.sum() - torch.sum(torch.exp(Lambda))

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
    A = adj_m
    A = torch.tensor(A)
    idx = torch.where(A > 0)
    count = A[idx[0],idx[1]]

    model = LSM(A=A, input_size=A.shape, latent_dim=2, sparse_i_idx= idx[0], sparse_j_idx=idx[1], count=count, sample_i_size = 1000, sample_j_size = 500)
#    B = model.optimizer(10)
    optimizer = optim.Adam(params=model.parameters(), lr=0.1)
    cum_loss = []
    for _ in range(10000):
        loss = -model.log_likelihood() / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        np.append(cum_loss,loss.item())
        print('Loss at the',_,'iteration:',loss.item())

    #Plot the whole lot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Blobs with 10k iteration optimization (Adam optimizer)')
    ax1.scatter(X1[:, 0], X1[:, 1], s= 30000 / len(X1), cmap="tab10", color="b")
    ax1.scatter(X2[:, 0], X2[:, 1], s= 30000 / len(X2), cmap="tab10", color="r")
    ax1.set_title("Before LSM")
    latent_zi = model.latent_zi.cpu().data.numpy()
    latent_zj = model.latent_zj.cpu().data.numpy()
    ax2.scatter(latent_zi[:, 0], latent_zi[:, 1], s= 30000 / len(latent_zi), cmap="tab10", color="b")
    ax2.scatter(latent_zj[:, 0], latent_zj[:, 1], s= 30000 / len(latent_zj), cmap="tab10", color="r")
    ax2.set_title('After LSM')
    plt.show()

    plt.plot(np.arange(10000),cum_loss)
    plt.title("Cumulative loss (lr=0.01)")
    plt.show()







