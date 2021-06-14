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

            #Determining AUC score and precision and recall
            auc_score = metrics.roc_auc_score(target.cpu().data.numpy(), rate.cpu().data.numpy())
            return auc_score, fpr, tpr

    #Implementing test log likelihood without mini batching
    def test_log_likelihood(self, A_test):
        with torch.no_grad():
            idx_test = torch.where(torch.isnan(A_test) == False)
            z_dist = (((torch.unsqueeze(self.latent_zi[idx_test[0]],1) - self.latent_zj[idx_test[1]] + 1e-06)**2).sum(-1))**0.5

            bias_matrix = torch.unsqueeze(self.beta[idx_test[0]],1) + self.gamma[idx_test[1]]
            Lambda = bias_matrix - z_dist
            LL_test = (A_test[idx_test[0],idx_test[1]] * Lambda).sum() - torch.sum(torch.exp(Lambda))
            return LL_test

if __name__ == "__main__":
    A = adj_m

    #Binarize data-set if True
    binarized = False
    link_pred = False
    cross_val = True

    for i in range(5):
        np.random.seed(i)
        torch.manual_seed(i)
        if binarized:
            A[A>0]= 1

        A = torch.tensor(A)

        #Sample test-set from multinomial distribution.
        if link_pred:
            A_shape = A.shape
            num_samples = 400000
            idx_i_test = torch.multinomial(input=torch.arange(0,float(A_shape[0])), num_samples=num_samples,
                                           replacement=True)
            idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples,
                                           replacement=True)
            A_test = A.detach().clone()
            A_test[:] = np.nan
            A_test[idx_i_test,idx_j_test] = A[idx_i_test,idx_j_test]
            A[idx_i_test,idx_j_test] = np.nan


        '''start = time.time()
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
        finish = time.time()
        print("time",finish-start)'''

        #Get the counts (only on train data)
        idx = torch.where((A > 0) & (torch.isnan(A) == False))
        count = A[idx[0],idx[1]]

        #Define the model with training data.
        #Cross-val loop validating 5 seeds;

        model = LSM(A=A, input_size=A.shape, latent_dim=2, sparse_i_idx= idx[0], sparse_j_idx=idx[1], count=count, sample_i_size = 1000, sample_j_size = 500)

        #Deine the optimizer.
        optimizer = optim.Adam(params=model.parameters(), lr=0.001)
        cum_loss = []
        cum_loss_test = []

        #Run iterations.
        iterations = 10
        for _ in range(iterations):
            loss = -model.log_likelihood() / model.input_size[0]
            if link_pred:
                loss_test = -model.test_log_likelihood(A_test) / model.input_size[0]
                cum_loss_test.append(loss_test.item())
                print('Test loss at the', _, 'iteration:', loss_test.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss.append(loss.item())
            print('Loss at the',_,'iteration:',loss.item())


        #Binary link-prediction enable and disable;
        if binarized:
            auc_score, fpr, tpr = model.link_prediction(A_test)

        beta = model.beta.cpu().data.numpy()
        gamma = model.gamma.cpu().data.numpy()
        latent_zi = model.latent_zi.cpu().data.numpy()
        latent_zj = model.latent_zj.cpu().data.numpy()
        np.savetxt(f"beta_{i}_link_pred_binary.csv", beta, delimiter=",")
        np.savetxt(f"gamma_{i}_link_pred_binary.csv", gamma, delimiter=",")
        np.savetxt(f"latent_zi_{i}_link_pred_binary.csv", latent_zi, delimiter=",")
        np.savetxt(f"latent_zj_{i}_link_pred_binary.csv", latent_zj, delimiter=",")
        np.savetxt(f"AUC_{i}_link_pred_binary.csv", auc_score, delimiter=",")
        np.savetxt(f"fpr_{i}_link_pred_binary.csv", fpr, delimiter=",")
        np.savetxt(f"tpr_{i}_link_pred_binary.csv", tpr, delimiter=",")
        np.savetxt(f"cum_loss_{i}_link_pred_binary.csv", cum_loss, delimiter=",")
