import os
import pandas as pd
import numpy as np

path = "Datasets/Single_cell"
os.chdir(path)
file = "raw_neuron_count_matrix.csv"
meta_data_file = "critical_period_neurons_metadata.csv"

df = pd.read_csv(file)
print(df)