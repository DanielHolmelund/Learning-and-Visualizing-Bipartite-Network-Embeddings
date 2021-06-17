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
            z_dist = (((self.latent_zi[idx_test[0]] - self.latent_zj[idx_test[1]] + 1e-06)**2).sum(-1))**0.5 #Unsqueeze eller ej?

            bias_matrix = self.beta[idx_test[0]] + self.gamma[idx_test[1]]
            Lambda = (bias_matrix - z_dist) * A_test[idx_test[0],idx_test[1]]
            LL_test = (A_test[idx_test[0],idx_test[1]] * Lambda).sum() - torch.sum(torch.exp(Lambda))
            return LL_test

if __name__ == "__main__":
    A = adj_m

    #Lists to obtain values for AUC, FPR, TPR and loss
    AUC_scores = []
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(5,5))

    train_loss = []
    test_loss = []


    #Binarize data-set if True
    binarized = False
    link_pred = True

    if binarized:
        A[A > 0] = 1

    A = torch.tensor(A)

    for i in range(5):
        np.random.seed(i)
        torch.manual_seed(i)

        #Sample test-set from multinomial distribution.
        if link_pred:
            A_shape = A.shape
            num_samples = 400000
            idx_i_test = torch.multinomial(input=torch.arange(0,float(A_shape[0])), num_samples=num_samples,
                                           replacement=True)
            idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples,
                                           replacement=True)
            A_test = A.detach().clone()
            A_test[:] = 0
            A_test[idx_i_test,idx_j_test] = A[idx_i_test,idx_j_test]
            A[idx_i_test,idx_j_test] = 0


        #Get the counts (only on train data)
        idx = torch.where((A > 0) & (torch.isnan(A) == False))
        count = A[idx[0],idx[1]]

        #Define the model with training data.
        #Cross-val loop validating 5 seeds;

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
                loss_test = -model.test_log_likelihood(A_test) / num_samples
                cum_loss_test.append(loss_test.item() / (A_shape[0]*A_shape[1] - num_samples))
                print('Test loss at the', _, 'iteration:', loss_test.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss.append(loss.item())
            print('Loss at the',_,'iteration:',loss.item())


        train_loss.append(cum_loss)
        test_loss.append(cum_loss_test)

        #Binary link-prediction enable and disable;
        if binarized:
            auc_score, fpr, tpr = model.link_prediction(A_test)

        AUC_scores.append(auc_score)
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)


        #beta = model.beta.cpu().data.numpy()
        #gamma = model.gamma.cpu().data.numpy()
        #latent_zi = model.latent_zi.cpu().data.numpy()
        #latent_zj = model.latent_zj.cpu().data.numpy()
        #np.savetxt(f"beta_{i}_link_pred_binary.csv", beta, delimiter=",")
        #np.savetxt(f"gamma_{i}_link_pred_binary.csv", gamma, delimiter=",")
        #np.savetxt(f"latent_zi_{i}_link_pred_binary.csv", latent_zi, delimiter=",")
        #np.savetxt(f"latent_zj_{i}_link_pred_binary.csv", latent_zj, delimiter=",")
        #np.savetxt(f"fpr_{i}_link_pred_binary.csv", fpr, delimiter=",")
        #np.savetxt(f"tpr_{i}_link_pred_binary.csv", tpr, delimiter=",")
        #np.savetxt(f"cum_loss_{i}_link_pred_binary.csv", cum_loss, delimiter=",")
        #np.savetxt(f"cum_loss_test_{i}_link_pred_binary.csv", cum_loss_test, delimiter=",")

    #np.savetxt(f"AUC_{i}_link_pred_binary.csv", AUC_scores, delimiter=",")



    #Plotting the average roc curve as a result of the cross-validation

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    #USing standard deviation as error bars
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b', label='Mean ROC-curve')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--', label='Random classifier')
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

    #Plotting the average loss based on the cross validation
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    mean_train_loss = train_loss.mean(axis=0)
    std_train_loss = train_loss.std(axis=0)
    mean_train_loss_upr = mean_train_loss + std_train_loss
    mean_train_loss_lwr = mean_train_loss - std_train_loss

    mean_test_loss = test_loss.mean(axis=0)
    std_test_loss = test_loss.std(axis=0)
    mean_test_loss_upr = mean_test_loss + std_test_loss
    mean_test_loss_lwr = mean_test_loss - std_test_loss


    plt.plot(np.arange(iterations), mean_train_loss, 'b', label='Mean training loss')
    plt.fill_between(np.arange(iterations), mean_train_loss_lwr, mean_train_loss_upr, color='b', alpha=0.3)
    plt.plot(np.arange(iterations), mean_test_loss, 'r', label='Mean test loss')
    plt.fill_between(np.arange(iterations), mean_test_loss_lwr, mean_test_loss_upr, color='r', alpha=0.3)
    plt.xlim([0, iterations])
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig('Average_loss.png')