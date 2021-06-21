import torch
import torch.optim as optim
import torch.nn as nn
from torch_sparse import spspmm
import pandas as pd
import numpy as np
# Creating dataset
from sklearn import metrics
import matplotlib.pyplot as plt

device = torch.device("cpu")
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
        sample_i_idx = torch.multinomial(self.sampling_i_weights, self.sample_i_size, replacement=False).to(device)
        sample_j_idx = torch.multinomial(self.sampling_j_weights, self.sample_j_size, replacement=False).to(device)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_i_translator = torch.cat([sample_i_idx.unsqueeze(0), sample_i_idx.unsqueeze(0)], 0).to(device)
        indices_j_translator = torch.cat([sample_j_idx.unsqueeze(0), sample_j_idx.unsqueeze(0)], 0).to(device)
        # adjacency matrix in edges format
        edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges, self.count.float(), indices_j_translator,
                                torch.ones(indices_j_translator.shape[1], device=device), self.input_size[0],
                                self.input_size[1],
                                self.input_size[1], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_i_translator, torch.ones(indices_i_translator.shape[1], device=device), indexC,
                                valueC,
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
        LL = (log_Lambda_links - torch.lgamma(valueC + 1)).sum() - torch.sum(torch.exp(self.Lambda))

        return LL


if __name__ == "__main__":

    idx_i = np.loadtxt("/work3/s194245/New_env/sample_data/data_sub_0.txt", delimiter=" ")
    idx_j = np.loadtxt("/work3/s194245/New_env/sample_data/data_sub_1.txt", delimiter=" ")
    value = np.loadtxt("/work3/s194245/New_env/sample_data/values_sub.txt", delimiter=" ")

    idx_i = torch.tensor(idx_i).to(device).long()
    idx_j = torch.tensor(idx_j).to(device).long()
    value = torch.tensor(value).to(device)

    learning_rate = 0.001  # Learning rate for adam

    # Define the model with training data.
    torch.manual_seed(0)

    dim = 3  # Chose the dimensionals for the embeddings

    model = LSM(input_size=(20526, 15743), latent_dim=dim, sparse_i_idx=idx_i, sparse_j_idx=idx_j, count=value,
                sample_i_size=2500, sample_j_size=2500).to(device)

    # Deine the optimizer.
    optimizer = optim.Adam(params=list(model.parameters()), lr=learning_rate)
    cum_loss = []

    # Run iterations.
    iterations = 1000000

    for _ in range(iterations):
        loss = -model.log_likelihood()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss.append(loss.item() / (model.sample_i_size * model.sample_j_size))
        if _ % 1000 == 0:
            torch.save(model.latent_zi.detach(), f"Embedding_{dim}d/latent_i_{_}")
            torch.save(model.latent_zj.detach(), f"Embedding_{dim}d/latent_j_{_}")
            torch.save(model.beta.detach(), f"Embedding_{dim}d/beta_{_}")
            torch.save(model.gamma.detach(), f"Embedding_{dim}d/gamma_{_}")
            np.savetxt(f"Embedding_{dim}d/cum_loss_{_}.txt", cum_loss, delimiter=" ")


