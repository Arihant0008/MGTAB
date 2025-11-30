import torch
from torch_geometric.data import Data

edge_index  = torch.load("edge_index.pt")
edge_type   = torch.load("edge_type.pt")
edge_weight = torch.load("edge_weight.pt")
features    = torch.load("features.pt")
labels      = torch.load("labels_bot.pt")

data = Data(
    x = features,
    edge_index = edge_index,
    edge_type = edge_type,
    edge_attr = edge_weight,
    y = labels
)

torch.save(data, "graph_data.pt")
print("\n✅ GRAPH READY:", data)
