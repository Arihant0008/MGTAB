import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import csv
import time

# ------------------------
# 1. LOAD GRAPH
# ------------------------
data = torch.load("graph_data.pt")

x = data.x
edge_index = data.edge_index
edge_attr = data.edge_attr            # [num_edges, 2]
edge_type = edge_attr[:, 0].long()    # relation IDs for RGCN
y = data.y

train_mask = data.train_mask
val_mask   = data.val_mask
test_mask  = data.test_mask

num_features = x.size(1)
num_classes = int(y.max().item() + 1)
num_relations = int(edge_type.max().item() + 1)

# ------------------------
# 2. CLASS IMBALANCE FIX
# ------------------------
num_real = (y == 0).sum().item()
num_bot  = (y == 1).sum().item()
weight_bot = num_real / num_bot      # ~2.7

class_weights = torch.tensor([1.0, weight_bot], dtype=torch.float32)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# ------------------------
# 3. DEFINE MODEL
# ------------------------
class RGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(num_features, 256, num_relations)
        self.conv2 = RGCNConv(256, num_classes, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

model = RGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val = 0
best_path = "best_rgcn.pt"

# ------------------------
# 4. LOG FILE SETUP
# ------------------------
log_path = "rgcn_training_log.csv"

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss", "Train_Acc", "Val_Acc", 
                     "Train_Bot_Recall", "Val_Bot_Recall", "Time(sec)"])

# ------------------------
# 5. TRAINING LOOP
# ------------------------
for epoch in range(1, 201):
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    out = model(x, edge_index, edge_type)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    # ------------------------
    # EVALUATION
    # ------------------------
    model.eval()
    with torch.no_grad():
        pred = out.argmax(dim=1)

        train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
        val_acc   = (pred[val_mask] == y[val_mask]).float().mean().item()

        train_recall = ((pred[train_mask] == 1) & (y[train_mask] == 1)).sum().item() / (y[train_mask] == 1).sum().item()
        val_recall   = ((pred[val_mask] == 1) & (y[val_mask] == 1)).sum().item() / (y[val_mask] == 1).sum().item()

    # UPDATE BEST MODEL
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), best_path)

    # LOG TO CSV
    elapsed = round(time.time() - start_time, 4)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss.item(), train_acc, val_acc,
                         train_recall, val_recall, elapsed])

    # PRINT EVERY 20 EPOCHS
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss={loss:.4f} | Train Acc={train_acc:.4f} | "
              f"Val Acc={val_acc:.4f} | Train Recall={train_recall:.4f} | "
              f"Val Recall={val_recall:.4f} | Time={elapsed}s")

print("\nBest Validation Accuracy:", best_val)

# ------------------------
# 6. TESTING
# ------------------------
model.load_state_dict(torch.load(best_path))
model.eval()

with torch.no_grad():
    out = model(x, edge_index, edge_type)
    pred = out.argmax(dim=1)

    test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
    bot_recall = ((pred[test_mask] == 1) & (y[test_mask] == 1)).sum().item() / (y[test_mask] == 1).sum().item()

print("Test Accuracy:", test_acc)
print("Bot Recall:", bot_recall)
print("\nLogs saved to:", log_path)
