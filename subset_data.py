import scanpy as sc
import numpy as np
import pandas as pd
import os

os.chdir("Datasets/Single_cell/dtudata")
neuron_sub_head = "neur.sub.h5ad"
neuron_sub = "neur.sub.mtx"

adata = sc.read(neuron_sub_head)

metadata = adata.obs
metadata2 = adata.var

df = pd.DataFrame(data=metadata)
df.to_csv("metadata_subdata.csv")

df2 = pd.DataFrame(data = metadata2)
df2.to_csv("var_metadata_subdata.csv")

