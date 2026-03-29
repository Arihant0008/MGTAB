import torch

data = torch.load("graph_data.pt")
print(data)
print("\nNode features:", data.x.shape)
print("Edges:", data.edge_index.shape)
print("Edge attributes:", data.edge_attr.shape)
print("Labels:", data.y.shape)
