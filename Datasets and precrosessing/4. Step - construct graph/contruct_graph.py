import torch
from torch_geometric.data import Data

edge_index  = torch.load("Dataset/edge_index.pt")
edge_type   = torch.load("Dataset/edge_type.pt")
edge_weight = torch.load("Dataset/edge_weight.pt")
features    = torch.load("Dataset/features.pt")
labels      = torch.load("Dataset/labels_bot.pt")

# Combine edge_type + edge_weight into edge_attr
edge_attr = torch.stack([edge_type, edge_weight], dim=1)

# Build final graph
data = Data(
    x = features,
    edge_index = edge_index,
    edge_attr = edge_attr,
    y = labels
)

torch.save(data, "graph_data.pt")
print("\n✅ GRAPH READY:", data)
