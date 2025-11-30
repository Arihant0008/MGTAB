import torch

print("Loading features...")
features = torch.load("features.pt")
print("Loaded!")

# print("Has NaN:", torch.isnan(features).any().item())
# print("Has Inf:", torch.isinf(features).any().item())
# print("Zero values:", (features == 0).sum().item())

print(type(features))
print(features.shape)
print(features[0][:20]) 
