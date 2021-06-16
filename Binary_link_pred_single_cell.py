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
    def __init__(self, input_size, latent_dim, sparse_i_idx, sparse_j_idx, count, sample_i_size, sample_j_size):
        super(LSM, self).__init__()
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

    def link_prediction(self, test_idx_i, test_idx_j, test_value):
        with torch.no_grad():
            # Distance measure (euclidian)
            z_pdist_test = (((self.latent_zi[test_idx_i] - self.latent_zj[test_idx_j] + 1e-06) ** 2).sum(-1)) ** 0.5

            # Add bias matrices
            logit_u_test = -z_pdist_test + self.beta[test_idx_i] + self.gamma[test_idx_j]

            # Get the rate
            rate = torch.exp(logit_u_test)

            # Create target (make sure its in the right order by indexing)
            target = test_value

            fpr, tpr, threshold = metrics.roc_curve(target.cpu().data.numpy(), rate.cpu().data.numpy())

            # Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())
            return auc_score, fpr, tpr

    # Implementing test log likelihood without mini batching
    def test_log_likelihood(self, test_idx_i, test_idx_j, test_value):
        with torch.no_grad():
            z_dist = (((self.latent_zi[test_idx_i] - self.latent_zj[test_idx_j] + 1e-06) ** 2).sum(-1)) ** 0.5

            bias_matrix = self.beta[test_idx_i] + self.gamma[test_idx_j]
            Lambda = (bias_matrix - z_dist) * test_value
            LL_test = (test_value * Lambda).sum() - torch.sum(torch.exp(Lambda))
            return LL_test


if __name__ == "__main__":
    #Train set:
    train_idx_i = np.loadtxt("data_train_0.txt", delimiter=" ")
    train_idx_j = np.loadtxt("data_train_1.txt", delimiter=" ")
    train_value = np.loadtxt("values_train.txt", delimiter=" ")

    train_idx_i = torch.tensor(train_idx_i).to(device).long()
    train_idx_j = torch.tensor(train_idx_j).to(device).long()
    train_value = torch.tensor(train_value).to(device)

    #Test set:
    test_idx_i = np.loadtxt("data_test_0.txt", delimiter=" ")
    test_idx_j = np.loadtxt("data_test_1.txt", delimiter=" ")
    test_value = np.loadtxt("values_test.txt", delimiter=" ")

    test_idx_i = torch.tensor(test_idx_i).to(device).long()
    test_idx_j = torch.tensor(test_idx_j).to(device).long()
    test_value = torch.tensor(test_value).to(device)

    #Binarize data-set
    test_value[:] = 1

    learning_rate = 0.01  # Learning rate for adam

    # Define the model with training data.
    # Cross-val loop validating 5 seeds;
    torch.manual_seed(0)

    model = LSM(input_size=(20526, 157430), latent_dim=2, sparse_i_idx=train_idx_i, sparse_j_idx=train_idx_j, count=train_value,
                sample_i_size=5000, sample_j_size=5000)

    #Deine the optimizer.
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    cum_loss_train = []
    cum_loss_test = []

    # Run iterations.
    iterations = 100000
    from copy import deepcopy

    for _ in range(iterations):
        loss = -model.log_likelihood()
        loss_test = -model.test_log_likelihood(test_idx_i, test_idx_j, test_value) / 323140818
        cum_loss_test.append(loss_test.item())
        auc_score, tpr, fpr = model.link_prediction(test_idx_i, test_idx_j, test_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss_train.append(loss.item() / (model.input_size[0]*model.input_size[1]-323140818))
        if _ % 1000 == 0:
            np.savetxt(f"binary_link_pred_output/latent_i_{_}.txt", deepcopy(model.latent_zi.detach().data), delimiter=" ")
            np.savetxt(f"binary_link_pred_output/latent_j_{_}.txt", deepcopy(model.latent_zj.detach().data), delimiter=" ")
            np.savetxt(f"binary_link_pred_output/beta_{_}.txt", deepcopy(model.beta.detach().data), delimiter=" ")
            np.savetxt(f"binary_link_pred_output/gamma_{_}.txt", deepcopy(model.gamma.detach().data), delimiter=" ")
            np.savetxt(f"binary_link_pred_output/cum_loss_train_{_}.txt", deepcopy(cum_loss_train.detach().data), delimiter=" ")
            np.savetxt(f"binary_link_pred_output/cum_loss_test_{_}.txt", deepcopy(cum_loss_test.detach().data), delimiter=" ")
            np.savetxt(f"binary_link_pred_output/auc_score_{_}.txt", auc_score, delimiter=" ")
            np.savetxt(f"binary_link_pred_output/tpr_{_}.txt", tpr, delimiter=" ")
            np.savetxt(f"binary_link_pred_output/fpr_{_}.txt", fpr, delimiter=" ")



