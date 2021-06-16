import numpy as np

A = np.loadtxt("/work3/s194245/raw_neuron_count_matrix.csv", delimiter = " ")

data = np.where(A > 0)
values = A[data[0], data[1]]

np.savetxt("data_0.txt", data[0], delimiter = " ")
np.savetxt("data_1.txt", data[1], delimiter = " ")
np.savetxt("values.txt", values, delimiter = " ")

