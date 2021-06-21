import numpy as np
import torch

A = np.loadtxt("/work3/s194245/raw_neuron_count_matrix.csv", delimiter = " ")


A_shape = (20526,157430)
num_samples = int(20000)
torch.manual_seed(0)
idx_j_test = torch.multinomial(input=torch.arange(0, float(A_shape[1])), num_samples=num_samples,
                               replacement=False)

A = torch.tensor(A)

sample = A[:,idx_j_test]

idx = torch.where(sample > 0)
value = sample[idx]

np.savetxt("data_sampled_0.txt", idx[0].numpy(), delimiter = " ")
np.savetxt("data_sampled_1.txt", idx[1].numpy(), delimiter = " ")
np.savetxt("values_sampled.txt", value.numpy(), delimiter = " ")


