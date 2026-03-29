import torch

files = [
    "Dataset/edge_index.pt",
    "Dataset/edge_type.pt",
    "Dataset/edge_weight.pt",
    "Dataset/features.pt",
    "Dataset/labels_bot.pt",
    "Dataset/labels_stance.pt"

]

for f in files:
    t = torch.load(f)
    print(f"{f} → shape: {None if not hasattr(t, 'shape') else t.shape}")
