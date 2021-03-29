import os
from scipy.io import mmread
import pandas as pd
"""
os.chdir("Datasets/Single_cell")

raw_data = mmread("critical_period_neurons_raw_counts.mtx")
A = raw_data.todense()
print(A[0:10,0:10])
df = pd.DataFrame(data=A.astype(float))
df.to_csv('outfile.csv', sep=' ', header=False, index=False)
"""
os.chdir("Datasets/divorce")

raw_data = mmread("divorce.mtx")
A = raw_data.todense()
B = A[:5,:4]
df_A = pd.DataFrame(data=A.astype(int))
df_B = pd.DataFrame(data=B.astype(int))
df_A.to_csv('outfile_tester.csv', sep=' ', header=False, index=False)
df_B.to_csv('outfile_tester2.csv', sep=' ', header=False, index=False)
