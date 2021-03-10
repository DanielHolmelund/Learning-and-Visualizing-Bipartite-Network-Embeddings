import torch
import os
from scipy.io import mmread
import torch.optim as optim
import torch.nn as nn
from Adjacency_matrix import Preprocessing

#Creating dataset

os.chdir('/Users/christiandjurhuus/PycharmProjects/Learning-and-Visualizing-Bipartite-Network-Embeddings/Datasets/divorce')

text_file = 'divorce.mtx'


#Loading data and making adjancency matrix
raw_data = mmread(text_file)
#print(raw_data)

A = raw_data.todense()
A = torch.tensor(A)
#print(A.shape)

class LSM(nn.Module):
    def __init__(self, B, input_size, latent_dim):
        super(LSM, self).__init__()
        self.A = B
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0]))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1]))

        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.latent_dim))

#        self.myparameters = nn.ParameterList([self.latent_zi, self.latent_zj, self.beta, self.gamma])
        #self.nn_layers = nn.Modulelist([self.latent_zi, self.latent_zj])


    def log_likelihood(self, A):

        #self.p_dist = torch.pairwise_distance(self.latent_zi, self.latent_zj, p=2)
        z_dist = (((torch.unsqueeze(self.latent_zi, 1) - self.latent_zj+1e-06)**2).sum(-1))**0.5
        bias_matrix = torch.unsqueeze(self.beta, 1) + self.gamma

        Lambda = bias_matrix - z_dist
        LL = (A*Lambda).sum() - torch.sum(torch.exp(Lambda))

        #LL = (torch.sum(torch.sparse.mm(A, Lambda)) - torch.sum(torch.exp(Lambda)))
        #LL = torch.sum(torch.sum(torch.sparse.mm(A, Lambda) - torch.exp(Lambda), dim=0), dim=1)
        #LL = torch.sum(torch.sparse.mm(A, Lambda) - torch.exp(Lambda), dim=0) * torch.sum(torch.sparse.mm(A, Lambda) - torch.exp(Lambda), dim=1)
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






