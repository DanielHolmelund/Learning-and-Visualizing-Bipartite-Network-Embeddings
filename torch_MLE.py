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
from sklearn import metrics

os.chdir('Datasets/divorce/')

text_file = 'divorce.mtx'


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
        self.z_dist = (((torch.unsqueeze(self.latent_zi[sample_i_idx], 1) - self.latent_zj[sample_j_idx]+1e-06)**2).sum(-1))**0.5
        bias_matrix = torch.unsqueeze(self.beta[sample_i_idx], 1) + self.gamma[sample_j_idx]
        self.Lambda = bias_matrix - self.z_dist
        z_dist_links = (((self.latent_zi[sparse_i_sample] - self.latent_zj[sparse_j_sample]+1e-06)**2).sum(-1))**0.5
        bias_links = self.beta[sparse_i_sample] + self.gamma[sparse_j_sample]
        log_Lambda_links = valueC*(bias_links - z_dist_links)
        LL = log_Lambda_links.sum() - torch.sum(torch.exp(self.Lambda))

        return LL

    def link_prediction(self, A_test):
        with torch.no_grad():
            #Create indexes for test-set relationships
            idx_test = torch.where(torch.isnan(A_test) == False)

            #Distance measure (euclidian)
            z_pdist_test = (((self.latent_zi[idx_test[0]] - self.latent_zj[idx_test[1]]+1e-06)**2).sum(-1))**0.5

            #Add bias matrices
            logit_u_test = -z_pdist_test + self.beta[idx_test[0]] + self.gamma[idx_test[1]]

            #Get the rate
            rate = torch.exp(logit_u_test)

            #Create target (make sure its in the right order by indexing)
            target = A_test[idx_test[0], idx_test[1]]

            fpr, tpr, threshold = metrics.roc_curve(target.cpu().data.numpy(), rate.cpu().data.numpy())
            plt.plot(fpr,tpr)
            plt.show()

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())
            print('AUC: %.3f' % auc_score)

    #Implementing test log likelihood without mini batching
    def test_log_likelihood(self, A_test):
        with torch.no_grad():
            idx_test = torch.where(torch.isnan(A_test) == False)
            z_dist = (((torch.unsqueeze(self.latent_zi[idx_test[0]], 1) - self.latent_zj[idx_test[1]] + 1e-06)**2).sum(-1))**0.5
            bias_matrix = torch.unsqueeze(self.beta[idx_test[0]], 1) + self.gamma[idx_test[1]]
            Lambda = bias_matrix - z_dist
            LL_test = (A * Lambda).sum() - torch.sum(torch.exp(Lambda))
            return LL_test

if __name__ == "__main__":
    A = adj_m

    #Binarize data-set if True
    binarized = False
    if binarized:
        A[A>0]= 1

    A = torch.tensor(A)

    #Separate train and test data with prob probability of moving a relationship to the test set. Enable and disable
    link_pred = True
    if link_pred:
        prob = 0.2
        np.random.seed(0)
        A_test = A.detach().clone()
        for i in range(A.size()[0]):
            for j in range(A.size()[1]):
                ran = np.random.rand(1)
                if ran < prob:
                    A[i,j] = np.nan
                else:
                    A_test[i,j] = np.nan


    #Get the counts (only on train data)
    idx = torch.where((A > 0) & (torch.isnan(A) == False))
    count = A[idx[0],idx[1]]

    #Define the model with training data.
    model = LSM(A=A, input_size=A.shape, latent_dim=2, sparse_i_idx= idx[0], sparse_j_idx=idx[1], count=count, sample_i_size = 1000, sample_j_size = 500)

    #Deine the optimizer.
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    cum_loss = []
    cum_loss_test = []

    #Run iterations.
    iterations = 100
    for _ in range(iterations):
        loss = -model.log_likelihood() / model.input_size[0]
        loss_test = -model.test_log_likelihood(A_test) / model.input_size[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss.append(loss.item())
        cum_loss_test.append(loss_test.item())
        print('Loss at the',_,'iteration:',loss.item())
        print('Test loss at the', _, 'iteration:', loss_test.item())

    #Binary link-prediction enable and disable;
    if binarized:
        model.link_prediction(A_test)

    #Plot the blobs data before and after LSM.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Blobs with 20k iteration optimization (Adam optimizer)')
    ax1.scatter(X1[:, 0], X1[:, 1], s= 30000 / len(X1), cmap="tab10", color="b")
    ax1.scatter(X2[:, 0], X2[:, 1], s= 30000 / len(X2), cmap="tab10", color="r")
    ax1.set_title("Before LSM")
    latent_zi = model.latent_zi.cpu().data.numpy()
    latent_zj = model.latent_zj.cpu().data.numpy()
    ax2.scatter(latent_zi[:, 0], latent_zi[:, 1], s= 30000 / len(latent_zi), cmap="tab10", color="b")
    ax2.scatter(latent_zj[:, 0], latent_zj[:, 1], s= 30000 / len(latent_zj), cmap="tab10", color="r")
    ax2.set_title('After LSM')
    plt.show()

    #Plot the loss at each iteration.
    plt.plot(np.arange(iterations), cum_loss)
    plt.plot(np.arange(iterations), cum_loss_test)
    plt.title("loss (lr=0.01)")
    plt.show()







