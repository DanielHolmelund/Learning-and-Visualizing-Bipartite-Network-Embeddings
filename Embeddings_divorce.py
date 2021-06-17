import torch
import torch.optim as optim
import torch.nn as nn
from torch_sparse import spspmm
import pandas as pd
import numpy as np
# Creating dataset
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import mmread

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class LSM(nn.Module):
    def __init__(self, input_size, latent_dim, sparse_i_idx, sparse_j_idx, count, sample_i_size, sample_j_size):
        super(LSM, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0], device=device))
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1], device=device))

        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim, device=device))
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.latent_dim, device=device))
        # Change sample weights for each partition
        self.sampling_i_weights = torch.ones(input_size[0]).to(device)
        self.sampling_j_weights = torch.ones(input_size[1]).to(device)
        # Change sample sizes for each partition
        self.sample_i_size = sample_i_size
        self.sample_j_size = sample_j_size

        self.sparse_i_idx = sparse_i_idx
        self.sparse_j_idx = sparse_j_idx

        self.count = count

        self.z_dist = 0
        self.Lambda = 0

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
        self.z_dist = (((torch.unsqueeze(self.latent_zi[sample_i_idx], 1) - self.latent_zj[
            sample_j_idx] + 1e-06) ** 2).sum(-1)) ** 0.5
        bias_matrix = torch.unsqueeze(self.beta[sample_i_idx], 1) + self.gamma[sample_j_idx]
        self.Lambda = bias_matrix - self.z_dist
        z_dist_links = (((self.latent_zi[sparse_i_sample] - self.latent_zj[sparse_j_sample] + 1e-06) ** 2).sum(
            -1)) ** 0.5
        bias_links = self.beta[sparse_i_sample] + self.gamma[sparse_j_sample]
        log_Lambda_links = valueC * (bias_links - z_dist_links)
        LL = log_Lambda_links.sum() - torch.sum(torch.exp(self.Lambda))

        return LL


if __name__ == "__main__":
    # Loading data and making adjancency matrix
    raw_data = mmread('Datasets/divorce/divorce.mtx')

    A = raw_data.todense()

    A = torch.tensor(A)

    idx = torch.where(A>0)

    value = A[idx]

    idx_i = idx[0]

    idx_j = idx[1]


    learning_rate = 0.01  # Learning rate for adam

    # Define the model with training data.
    torch.manual_seed(0)

    model = LSM(input_size=(50, 9), latent_dim=2, sparse_i_idx=idx_i, sparse_j_idx=idx_j, count=value,
                sample_i_size=30, sample_j_size=6).to(device)

    # Deine the optimizer.
    optimizer = optim.Adam(params=list(model.parameters()), lr=learning_rate)
    cum_loss = []

    # Run iterations.
    iterations = 100000
    from copy import deepcopy

    for _ in range(iterations):
        loss = -model.log_likelihood()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss.append(loss.item() / (model.sample_i_size * model.sample_j_size))
        print("hey")
        if _ % 1000 == 0:
            np.savetxt(f"model_output/latent_i_{_}.txt", model.latent_zi.detach().cpu().data, delimiter=" ")
            np.savetxt(f"model_output/latent_j_{_}.txt", model.latent_zj.detach().cpu().data, delimiter=" ")
            np.savetxt(f"model_output/beta_{_}.txt", model.beta.detach().cpu().data, delimiter=" ")
            np.savetxt(f"model_output/gamma_{_}.txt", model.gamma.detach().cpu().data, delimiter=" ")
            np.savetxt(f"model_output/cum_loss_{_}.txt", cum_loss, delimiter=" ")


