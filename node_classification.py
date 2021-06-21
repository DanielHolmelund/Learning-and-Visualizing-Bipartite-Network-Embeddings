from LabelEncoder import *
import sklearn.linear_model as lin
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
np.random.seed(42)

### Get the metadata

df = pd.read_csv("C:/Users/Daniel/Documents/DTU/Reinforcement/Learning-and-Visualizing-Bipartite-Network-Embeddings/Datasets/Single_cell/dtudata/metadata_subdata.csv")
#cate = Categorization(df)
idx_seurat = df.columns.get_loc("age")
#labels = cate.encoder() # label encoding
labels = df.loc[:, "age"].values # Only take the relevant labels
### remove the first row
#labels = labels[1:]

### Get the embeddings
#embeddings_filename = "C:/Users/Daniel/Documents/DTU/Reinforcement/Learning-and-Visualizing-Bipartite-Network-Embeddings/results/embeddings_50d/latent_i_5000"
embeddings_filename = "C:/Users/Daniel/Documents/DTU/Reinforcement/Learning-and-Visualizing-Bipartite-Network-Embeddings/results/embeddings_50d/latent_j_5000"
#df = pd.read_csv(embeddings_filename)
#embeddings = df.to_numpy()
embeddings = torch.load(embeddings_filename).data.cpu().data.numpy()


X = embeddings
y = labels
y = y.astype('int')

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
LR = model.fit(train_X, train_y)
pred_y = model.predict(test_X)
accuracy_LR = metrics.accuracy_score(test_y, pred_y)
print(f"Accuracy score: {accuracy_LR}")


model = RandomForestClassifier()
RF = model.fit(train_X, train_y)
pred_y = model.predict(test_X)
accuracy_RF = metrics.accuracy_score(test_y, pred_y)
print(f"Accuracy score: {accuracy_RF}")

with open('metrics_node.txt', 'w') as f:

    f.write(f"Accuracy score: {accuracy_LR} for LR")
    f.write("\n")
    f.write(f"Accuracy score: {accuracy_RF} for random forest")
