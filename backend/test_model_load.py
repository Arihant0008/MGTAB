"""Quick test: verify best_rgcn.pt loads into our RGCN architecture."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(788, 256, 7)
        self.conv2 = RGCNConv(256, 2, 7)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

model = RGCN()
sd = torch.load("../Datasets and precrosessing/best_rgcn.pt", map_location="cpu", weights_only=True)
model.load_state_dict(sd)
model.eval()

print("✓ Model loaded successfully!")
for k, v in sd.items():
    print(f"  {k}: {v.shape}")

# Quick inference test with dummy data
x = torch.randn(5, 788)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
edge_type = torch.tensor([0, 1, 2], dtype=torch.long)

with torch.no_grad():
    out = model(x, edge_index, edge_type)
    probs = F.softmax(out, dim=1)
    preds = probs.argmax(dim=1)

print(f"\n✓ Inference works! Output shape: {out.shape}")
print(f"  Predictions: {preds.tolist()}")
print(f"  Probabilities: {probs.tolist()}")
