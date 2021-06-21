import numpy as np
import pandas as pd

data = "Datasets/Single_cell/data_sampled_0.txt"
data = np.loadtxt(data)

length = np.size(data)
print(length)


