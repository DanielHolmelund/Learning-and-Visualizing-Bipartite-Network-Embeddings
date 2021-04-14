
from sklearn import preprocessing
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
os.chdir("Datasets/Single_cell")

df = pd.read_csv("critical_period_neurons_metadata.csv")

class Categorization():
    def __init__(self, df):
        self.df = df
        self.X = df.iloc[:, :].values

    def get_type_index(self):
        idx = []
        i = 0
        for column in self.df:
            dataTypeObj = self.df.dtypes[column]
            if dataTypeObj == np.object_:
                idx.append(i)
            if dataTypeObj == np.bool_:
                idx.append(i)
            i += 1
        return idx

    def encoder(self):
        labelEncoder = LabelEncoder()
        indices = self.get_type_index()
        for col in range(len(df.columns)):
            if col in indices:
                self.X[:, col] = labelEncoder.fit_transform(self.X[:, col])
        return self.X

if __name__ == "__main__":
    cate = Categorization(df)
    data = cate.encoder()
    print(data)



