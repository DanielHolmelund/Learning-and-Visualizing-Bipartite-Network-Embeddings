import numpy as np
import torch

A = np.loadtxt("/work3/s194245/raw_neuron_count_matrix.csv", delimiter = " ")

A_shape = (20526,157430)
num_samples = 323140818
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

np.savetxt("data_train_0.txt", train_data_idx[0], delimiter = " ")
np.savetxt("data_train_1.txt", train_data_idx[1], delimiter = " ")
np.savetxt("values_train.txt", values_train, delimiter = " ")

np.savetxt("data_test_0.txt", idx_i_test, delimiter = " ")
np.savetxt("data_test_1.txt", idx_j_test, delimiter = " ")
np.savetxt("values_test.txt", value_test, delimiter = " ")
