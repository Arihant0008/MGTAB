import torch
from torch_geometric.data import Data

data = torch.load("graph_data.pt")

num_nodes = data.num_nodes

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

perm = torch.randperm(num_nodes)
train_end = int(train_ratio * num_nodes)
val_end = int((train_ratio + val_ratio) * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[perm[:train_end]] = True
val_mask[perm[train_end:val_end]] = True
test_mask[perm[val_end:]] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

torch.save(data, "graph_data.pt")
print("Masks added:", data)
