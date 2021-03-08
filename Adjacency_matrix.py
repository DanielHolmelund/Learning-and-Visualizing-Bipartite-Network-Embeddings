import torch
# Transform the Biadjacency matrix to a square adjacency for bipartite graphs.
def From_Biadjacency_To_Adjacency(B):
    r, c = list(B.size())
    Adjacency_Matrix = torch.cat((torch.cat((torch.zeros(r,r), B),1),
                                  torch.cat((torch.transpose(B,0,1), torch.zeros(c,c)),1)),0)
    return Adjacency_Matrix

if __name__ == "__main__":
    B = torch.rand(50,9)
    print(From_Biadjacency_To_Adjacency(B))
    print(From_Biadjacency_To_Adjacency(B).size())