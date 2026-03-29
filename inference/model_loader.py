"""
model_loader.py — Loads the RGCN model and graph data at startup.
Keeps everything in memory so inference is fast.
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so it works from any cwd
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Datasets and precrosessing")

GRAPH_DATA_PATH = os.path.join(DATA_DIR, "graph_data.pt")
MODEL_WEIGHTS_PATH = os.path.join(DATA_DIR, "best_rgcn.pt")

# ---------------------------------------------------------------------------
# Load graph data once
# ---------------------------------------------------------------------------
print(f"[model_loader] Loading graph data from {GRAPH_DATA_PATH} ...")
graph_data = torch.load(GRAPH_DATA_PATH, weights_only=False, map_location="cpu")

x = graph_data.x                          # node features  [N, F]
edge_index = graph_data.edge_index         # edge connectivity [2, E]
edge_attr = graph_data.edge_attr           # [E, 2]
edge_type = edge_attr[:, 0].long()         # relation IDs for RGCN
y = graph_data.y                           # ground-truth labels
test_mask = graph_data.test_mask

NUM_FEATURES = x.size(1)
NUM_CLASSES = int(y.max().item() + 1)
NUM_RELATIONS = int(edge_type.max().item() + 1)
NUM_NODES = x.size(0)

print(f"[model_loader] Graph loaded — {NUM_NODES} nodes, {edge_index.size(1)} edges, "
      f"{NUM_FEATURES} features, {NUM_RELATIONS} relations, {NUM_CLASSES} classes")

# ---------------------------------------------------------------------------
# Define RGCN architecture (must match training code)
# ---------------------------------------------------------------------------
class RGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(NUM_FEATURES, 256, NUM_RELATIONS)
        self.conv2 = RGCNConv(256, NUM_CLASSES, NUM_RELATIONS)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

# ---------------------------------------------------------------------------
# Load trained weights
# ---------------------------------------------------------------------------
print(f"[model_loader] Loading model weights from {MODEL_WEIGHTS_PATH} ...")
model = RGCN()
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True, map_location="cpu"))
model.eval()
print("[model_loader] RGCN model loaded and set to eval mode ✓")

# ---------------------------------------------------------------------------
# Pre-compute all predictions (the graph fits in memory)
# ---------------------------------------------------------------------------
print("[model_loader] Running full-graph forward pass ...")
with torch.no_grad():
    logits = model(x, edge_index, edge_type)
    probabilities = F.softmax(logits, dim=1)       # [N, 2]
    predictions = logits.argmax(dim=1)              # [N]

print("[model_loader] Pre-computation complete ✓")

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "human", 1: "bot"}

def get_prediction(node_index: int) -> dict:
    """Return prediction for a single node index."""
    if node_index < 0 or node_index >= NUM_NODES:
        return None

    pred_class = predictions[node_index].item()
    confidence = probabilities[node_index][pred_class].item()

    return {
        "nodeIndex": node_index,
        "prediction": LABEL_MAP[pred_class],
        "confidence": round(confidence, 4),
        "groundTruth": LABEL_MAP[y[node_index].item()] if y[node_index].item() in LABEL_MAP else "unknown",
    }

def get_total_nodes() -> int:
    return NUM_NODES

def get_dataset_stats() -> dict:
    num_bots = int((y == 1).sum().item())
    num_humans = int((y == 0).sum().item())
    return {
        "totalNodes": NUM_NODES,
        "totalEdges": int(edge_index.size(1)),
        "numBots": num_bots,
        "numHumans": num_humans,
        "numFeatures": NUM_FEATURES,
        "numRelations": NUM_RELATIONS,
    }
