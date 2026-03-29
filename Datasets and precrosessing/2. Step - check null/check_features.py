import torch

edge_index  = torch.load("Dataset/edge_index.pt")
edge_type   = torch.load("Dataset/edge_type.pt")
edge_weight = torch.load("Dataset/edge_weight.pt")
features    = torch.load("Dataset/features.pt")
labels      = torch.load("Dataset/labels_bot.pt")

print("---- FEATURE CHECK ----")
print("NaN features:", torch.isnan(features).sum())
print("Inf features:", torch.isinf(features).sum())
print("Zero feature rows:", (features.sum(dim=1) == 0).sum())

print("\n---- LABEL CHECK ----")
print("NaN labels:", torch.isnan(labels).sum())

print("\n---- EDGE INDEX CHECK ----")
num_nodes = features.shape[0]
print("NaN edge_index:", torch.isnan(edge_index).sum())
print("Negative edges:", (edge_index < 0).sum())
print("Invalid edges:", ((edge_index >= num_nodes)).sum())

print("\n---- EDGE TYPE CHECK ----")
print("NaN edge_type:", torch.isnan(edge_type).sum())
print("Min type:", edge_type.min().item())
print("Max type:", edge_type.max().item())

print("\n---- EDGE WEIGHT CHECK ----")
print("NaN weights:", torch.isnan(edge_weight).sum())
print("Negative weights:", (edge_weight < 0).sum())
print("Zero weights:", (edge_weight == 0).sum())
