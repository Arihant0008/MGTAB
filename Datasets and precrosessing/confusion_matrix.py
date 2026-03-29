import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------
# 1. LOAD GRAPH + MODEL
# ----------------------
data = torch.load("graph_data.pt", weights_only=False)

x = data.x
edge_index = data.edge_index

# 🔥 FIX: Use edge_attr to get edge_type
edge_attr = data.edge_attr  # [num_edges, 2]
edge_type = edge_attr[:, 0].long()

y = data.y
test_mask = data.test_mask

num_features = x.size(1)
num_classes = int(y.max().item() + 1)
num_rel = int(edge_type.max().item() + 1)

class RGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(num_features, 256, num_rel)
        self.conv2 = RGCNConv(256, num_classes, num_rel)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = F.dropout(x, 0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

model = RGCN()
model.load_state_dict(torch.load("best_rgcn.pt"))
model.eval()

# ----------------------
# 2. INFERENCE
# ----------------------
with torch.no_grad():
    logits = model(x, edge_index, edge_type)
    preds = logits.argmax(dim=1)

# ----------------------
# 3. GET TEST LABELS
# ----------------------
y_true = y[test_mask].cpu().numpy()
y_pred = preds[test_mask].cpu().numpy()

# ----------------------
# 4. CONFUSION MATRIX
# ----------------------
cm = confusion_matrix(y_true, y_pred)
print("\n=== CONFUSION MATRIX (RGCN) ===")
print(cm)

# ----------------------
# 5. CLASSIFICATION REPORT
# ----------------------
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=["Human", "Bot"]))
