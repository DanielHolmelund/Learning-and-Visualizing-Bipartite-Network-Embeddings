import torch
import torch.optim as optim
import torch.nn as nn
from torch_sparse import spspmm
import pandas as pd
import numpy as np
#Creating dataset
from blobs import *
from sklearn import metrics


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
        LL = (log_Lambda_links-torch.lgamma(valueC+1)).sum() - torch.sum(torch.exp(self.Lambda))

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

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())
            return auc_score, fpr, tpr

    #Implementing test log likelihood without mini batching
    def test_log_likelihood(self, A_test):
        with torch.no_grad():
            idx_test = torch.where(torch.isnan(A_test) == False)
            z_dist = (((self.latent_zi[idx_test[0]] - self.latent_zj[idx_test[1]] + 1e-06)**2).sum(-1))**0.5 #Unsqueeze eller ej?

            bias_matrix = self.beta[idx_test[0]] + self.gamma[idx_test[1]]
            Lambda = bias_matrix - z_dist
            LL_test = (A_test[idx_test[0],idx_test[1]] * Lambda).sum() - torch.sum(torch.exp(Lambda))
            return LL_test

if __name__ == "__main__":
    A = adj_m


    train_loss = []
    test_loss = []


    #Binarize data-set if True
    binarized = False
    link_pred = False

    if binarized:
        A[A > 0] = 1

    A = torch.tensor(A)
    

    #Sample test-set from multinomial distribution.
    if link_pred:
        A_shape = A.shape
        num_samples = 400000
        idx_i_test = torch.multinomial(input=torch.arange(0,float(A_shape[0])), num_samples=num_samples, replacement=True)
        idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples, replacement=True)
        A_test = A.detach().clone()
        A_test[:] = np.nan
        A_test[idx_i_test,idx_j_test] = A[idx_i_test,idx_j_test]
        A[idx_i_test,idx_j_test] = np.nan


        #Get the counts (only on train data)
        idx = torch.where((A > 0) & (torch.isnan(A) == False))
        count = A[idx[0],idx[1]]


    # Get the counts (only on train data)
    idx = torch.where((A > 0) & (torch.isnan(A) == False))
    count = A[idx[0], idx[1]]

    #Define the model with training data.

    model = LSM(A=A, input_size=A.shape, latent_dim=2, sparse_i_idx= idx[0], sparse_j_idx=idx[1], count=count, sample_i_size = 1000, sample_j_size = 500)

#Deine the optimizer.
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    cum_loss = []
    cum_loss_test = []

#Run iterations.
    iterations = 20000
    for _ in range(iterations):
        loss = -model.log_likelihood()
        if link_pred:
            loss_test = -model.test_log_likelihood(A_test)
            cum_loss_test.append(loss_test.item())
            # print('Test loss at the', _, 'iteration:', loss_test.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss.append(loss.item() / (model.sample_i_size * model.sample_j_size))
        # print('Loss at the',_,'iteration:',loss.item())

#Plot the blobs data before and after LSM.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X1[:, 0], X1[:, 1], s= 30000 / len(X1), cmap="tab10", color="b")
    ax1.scatter(X2[:, 0], X2[:, 1], s= 30000 / len(X2), cmap="tab10", color="r")
    ax1.set_title("Original embeddings")
    latent_zi = model.latent_zi.cpu().data.numpy()
    latent_zj = model.latent_zj.cpu().data.numpy()
    ax2.scatter(latent_zi[:, 0], latent_zi[:, 1], s= 30000 / len(latent_zi), cmap="tab10", color="b", label='first partition')
    ax2.scatter(latent_zj[:, 0], latent_zj[:, 1], s= 30000 / len(latent_zj), cmap="tab10", color="r", label='second partition')
    ax2.set_title('Recovered embeddings')
    #plt.legend()
    plt.savefig('Dense_recover_sampling.png')
    plt.clf()


    #Plot the loss at each iteration.
    plt.plot(np.arange(iterations), cum_loss, label='Train')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid()
    #plt.legend()
    plt.savefig('Dense_loss_sampling.png')

