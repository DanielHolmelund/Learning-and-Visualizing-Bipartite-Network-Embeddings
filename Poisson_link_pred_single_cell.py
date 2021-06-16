import torch
import torch.optim as optim
import torch.nn as nn
from torch_sparse import spspmm
import pandas as pd
import numpy as np
# Creating dataset
from sklearn import metrics
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class LSM(nn.Module):
    def __init__(self, A, input_size, latent_dim, sparse_i_idx, sparse_j_idx, count, sample_i_size, sample_j_size):
        super(LSM, self).__init__()
        self.A = A
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.beta = torch.nn.Parameter(torch.randn(self.input_size[0])).to(device)
        self.gamma = torch.nn.Parameter(torch.randn(self.input_size[1])).to(device)

        self.latent_zi = torch.nn.Parameter(torch.randn(self.input_size[0], self.latent_dim)).to(device)
        self.latent_zj = torch.nn.Parameter(torch.randn(self.input_size[1], self.latent_dim)).to(device)
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

    def link_prediction(self, A_test):
        with torch.no_grad():
            # Create indexes for test-set relationships
            idx_test = torch.where(torch.isnan(A_test) == False)

            # Distance measure (euclidian)
            z_pdist_test = (((self.latent_zi[idx_test[0]] - self.latent_zj[idx_test[1]] + 1e-06) ** 2).sum(-1)) ** 0.5

            # Add bias matrices
            logit_u_test = -z_pdist_test + self.beta[idx_test[0]] + self.gamma[idx_test[1]]

            # Get the rate
            rate = torch.exp(logit_u_test)

            # Create target (make sure its in the right order by indexing)
            target = A_test[idx_test[0], idx_test[1]]

            fpr, tpr, threshold = metrics.roc_curve(target.cpu().data.numpy(), rate.cpu().data.numpy())

            # Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())
            return auc_score, fpr, tpr

    # Implementing test log likelihood without mini batching
    def test_log_likelihood(self, A_test):
        with torch.no_grad():
            idx_test = torch.where(torch.isnan(A_test) == False)
            z_dist = (((self.latent_zi[idx_test[0]] - self.latent_zj[idx_test[1]] + 1e-06) ** 2).sum(-1)) ** 0.5

            bias_matrix = self.beta[idx_test[0]] + self.gamma[idx_test[1]]
            Lambda = (bias_matrix - z_dist)
            LL_test = (A_test[idx_test[0], idx_test[1]] * Lambda).sum() - torch.sum(torch.exp(Lambda))
            return LL_test


if __name__ == "__main__":

    idx_i = np.loadtxt("data_0.txt", delimiter=" ")
    idx_j = np.loadtxt("data_1.txt", delimiter=" ")
    value = np.loadtxt("values.txt", delimiter=" ")

    idx_i = torch.tensor(idx_i).to(device).long()
    idx_j = torch.tensor(idx_j).to(device).long()
    value = torch.tensor(idx_j).to(device)

    # Binarize data-set if True
    binarized = False
    link_pred = False

    # Lists to obtain values for AUC, FPR, TPR and loss
    AUC_scores = []
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(5, 5))

    train_loss = []
    test_loss = []

    learning_rate = 0.01  # Learning rate for adam
    if binarized:
        A[A > 0] = 1


    # Define the model with training data.
    # Cross-val loop validating 5 seeds;

    model = LSM(input_size=(20526, 157430), latent_dim=2, sparse_i_idx=idx_i, sparse_j_idx=idx_j, count=value,
                sample_i_size=5000, sample_j_size=5000)

    # Deine the optimizer.
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    cum_loss = []
    cum_loss_test = []

    # Run iterations.
    iterations = 100000
    from copy import deepcopy

    for _ in range(iterations):
        loss = -model.log_likelihood()
        # if link_pred:
        # loss_test = -model.test_log_likelihood(A_test)
        # cum_loss_test.append(loss_test.item())
        # print('Test loss at the', _, 'iteration:', loss_test.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss.append(loss.item() / (model.sample_i_size * model.sample_j_size))
        if _ % 1000 == 0:
            np.savetxt(f"model_output/latent_i_{_}.txt", deepcopy(model.latent_zi.detach().data), delimiter=" ")
            np.savetxt(f"model_output/latent_j_{_}.txt", deepcopy(model.latent_zj.detach().data), delimiter=" ")
            np.savetxt(f"model_output/beta_{_}.txt", deepcopy(model.beta.detach().data), delimiter=" ")
            np.savetxt(f"model_output/gamma_{_}.txt", deepcopy(model.gamma.detach().data), delimiter=" ")
            np.savetxt(f"model_output/cum_loss_{_}.txt", deepcopy(model.cum_loss.detach().data), delimiter=" ")
        # print('Loss at the',_,'iteration:',loss.item())


