import torch
import torch.optim as optim
import torch.nn as nn
from torch_sparse import spspmm
import pandas as pd
import numpy as np
# Creating dataset
from sklearn import metrics
import matplotlib.pyplot as plt
from blobs import adj_m

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
        #self.sampling_i_weights[test_idx_i] = 0 #dont sample test set :D
        self.sampling_j_weights = torch.ones(input_size[1]).to(device)
        #self.sampling_j_weights[test_idx_j] = 0 #same
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
                                torch.ones(indices_j_translator.shape[1],device=device), self.input_size[0], self.input_size[1],
                                self.input_size[1], coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_i_translator, torch.ones(indices_i_translator.shape[1],device=device), indexC, valueC,
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
        LL = (log_Lambda_links-torch.lgamma(valueC+1)).sum() - torch.sum(torch.exp(self.Lambda))

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
            Lambda = (bias_matrix - z_dist)
            LL_test = ((test_value * Lambda) - (torch.lgamma(test_value+1))).sum() - torch.sum(torch.exp(Lambda))
            return LL_test


if __name__ == "__main__":
    A = adj_m


    '''idx = torch.where(A > 0)

    value = A[idx]

    idx_i = idx[0]

    idx_j = idx[1]'''

    #Lists to store obtained losses
    train_loss = []
    test_loss = []

    A_shape = (2000, 1000)
    num_samples = 200000
    idx_i_test = torch.multinomial(input=torch.arange(0, float(A_shape[0])), num_samples=num_samples,
                                   replacement=True)
    idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples,
                                   replacement=True)

    A = torch.tensor(A)

    value_test = A[idx_i_test, idx_j_test].numpy()

    A[idx_i_test, idx_j_test] = 0

    # Train data
    train_data_idx = torch.where(A > 0)
    values_train = A[train_data_idx[0], train_data_idx[1]].numpy()

    train_idx_i = train_data_idx[0]
    train_idx_j = train_data_idx[1]
    train_value = torch.tensor(values_train)

    test_idx_i = idx_i_test
    test_idx_j = idx_j_test
    test_value = torch.tensor(value_test)

    test_value[test_value > 0] = 1

    learning_rate = 0.01  # Learning rate for adam

    #For binary link prediction
    #Lists to obtain values for AUC, FPR, TPR and loss
    AUC_scores = []
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(5,5))


    # Define the model with training data.
    # Cross-val loop validating 5 seeds;
    for i in range(5):
        np.random.seed(i)
        torch.manual_seed(i)

        model = LSM(input_size=(2000, 1000), latent_dim=2, sparse_i_idx=train_idx_i, sparse_j_idx=train_idx_j, count=train_value,
                    sample_i_size=2000, sample_j_size=1000).to(device)

        #Deine the optimizer.
        optimizer = optim.Adam(params=list(model.parameters()), lr=learning_rate)
        cum_loss_train = []
        cum_loss_test = []

        # Run iterations.
        iterations = 10

        for _ in range(iterations):
            loss = -model.log_likelihood()
            loss_test = -model.test_log_likelihood(test_idx_i, test_idx_j, test_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss_test.append(loss_test.item() / num_samples)
            cum_loss_train.append(loss.item() / ((model.sample_i_size*model.sample_j_size)))


        train_loss.append(cum_loss_train)
        test_loss.append(cum_loss_test)

        # Binary link-prediction:
        auc_score, fpr, tpr = model.link_prediction(test_idx_i, test_idx_j, test_value)

        AUC_scores.append(auc_score)
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    print(np.mean(AUC_scores) + np.array([-1,1]) * 1.96 * np.sqrt(np.var(AUC_scores)/len(AUC_scores)))
    print(np.mean(AUC_scores))
    print( 1.96 * np.sqrt(np.var(AUC_scores)/len(AUC_scores)))

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    # USing standard deviation as error bars
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b', label='Mean ROC-curve')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1], 'r--', label='Random classifier')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig('Average_ROC_curve.png')
    plt.clf()


