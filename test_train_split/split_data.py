import torch
import torch_geometric
from torch_geometric.data import Data  # important import

# Allow PyG Data object to be loaded safely
torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

# Now load graph
data = torch.load("graph_data.pt", weights_only=False)

num_nodes = data.x.size(0)

# 80 / 10 / 10 split (fixed + reproducible)
torch.manual_seed(42)
idx = torch.randperm(num_nodes)

train_size = int(0.8 * num_nodes)
val_size   = int(0.1 * num_nodes)

train_idx = idx[:train_size]
val_idx   = idx[train_size:train_size + val_size]
test_idx  = idx[train_size + val_size:]

train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[train_idx] = True
val_mask   = torch.zeros(num_nodes, dtype=torch.bool); val_mask[val_idx] = True
test_mask  = torch.zeros(num_nodes, dtype=torch.bool); test_mask[test_idx] = True

torch.save(train_mask, "train_mask.pt")
torch.save(val_mask, "val_mask.pt")
torch.save(test_mask, "test_mask.pt")

print("✅ Train/Val/Test masks saved!")
print("Train samples:", train_mask.sum().item())
print("Val samples:", val_mask.sum().item())
print("Test samples:", test_mask.sum().item())
