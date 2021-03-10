import torch
import os
from scipy.io import mmread
import torch.optim
from Adjacency_matrix import Preprocessing

#Creating dataset

os.chdir('/Users/christiandjurhuus/PycharmProjects/Learning-and-Visualizing-Bipartite-Network-Embeddings/Datasets/divorce')

text_file = 'divorce.mtx'


#Loading data and making adjancency matrix
raw_data = mmread(text_file)
#print(raw_data)

A = raw_data.todense()
print(A.shape)

class LSM():
    def __init__(self, A, input_size, latent_dim):
        self.A = Preprocessing.From_Biadjacency_To_Adjacency(A)
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn(self.input_size,device=device))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size,device=device))

        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size, self.latent_dim, device=device))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size, self.latent_dim, device=device))



    def log_likelihood(self, A):

        #self.p_dist = torch.pairwise_distance(self.latent_zi, self.latent_zj, p=2)
        z_dist = (((torch.unsqueeze(self.latent_zi, 1) - self.latent_zj)**2).sum(-1))**0.5

        Lambda = torch.unsqueeze(self.beta, 1) + self.gamma - abs(z_dist)

        LL += torch.unsqueeze(torch.sparse.mm(A, Lambda),1).sum() - torch.sum(torch.exp(Lambda))
        #LL = (torch.sum(torch.sparse.mm(A, Lambda)) - torch.sum(torch.exp(Lambda)))
        #LL = torch.sum(torch.sum(torch.sparse.mm(A, Lambda) - torch.exp(Lambda), dim=0), dim=1)
        #LL = torch.sum(torch.sparse.mm(A, Lambda) - torch.exp(Lambda), dim=0) * torch.sum(torch.sparse.mm(A, Lambda) - torch.exp(Lambda), dim=1)

    def optimizer(self, iterations):
        # Implements stochastic gradient descent (optionally with momentum). Nesterov momentum

        for _ in range(iterations):
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            loss = -model.log_likelihood(self.A)/ self.input_size
            loss.backward()
            optimizer.step()
        return loss

if __name__ == "__main__":
    model = LSM(A=A, input_size=A.shape[0], latent_dim=2)
    B = model.optimizer(10)






