import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.nn import RGCNConv

# Define the RGCN model (copied from train_bot_rgcn.py)
class RGCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hid_dim, num_relations)
        self.conv2 = RGCNConv(hid_dim, out_dim, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = F.dropout(x, 0.5, self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

# Placeholder for enhance_features function
def enhance_features(graph_data):
    # Assuming no additional feature enhancement is required
    return graph_data

def evaluate_model(model_path, graph_path, test_mask_path, threshold=0.5):
    # Load the graph data and test mask
    graph_data = torch.load(graph_path)
    test_mask = torch.load(test_mask_path)

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGCN(788, 256, 2, graph_data.edge_type.max().item() + 1).to(device)  # Updated input feature size to 788
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = model.state_dict()

    # Filter out incompatible layers
    filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    model_state_dict.update(filtered_state_dict)

    # Load the filtered state dict
    model.load_state_dict(model_state_dict)

    model.eval()

    # Move data to device
    graph_data = graph_data.to(device)
    test_mask = test_mask.to(device)
    test_labels = graph_data.y[test_mask].to(device)

    # Enhance features before testing
    graph_data = enhance_features(graph_data)

    # Perform inference on the test set
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_type)
        probs = F.softmax(out, dim=1)  # Get probabilities
        pred = (probs[:, 1] >= threshold).long()  # Apply threshold for class 1 (bots)

    # Extract test predictions
    test_preds = pred[test_mask]

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels.cpu(), test_preds.cpu())
    precision = precision_score(test_labels.cpu(), test_preds.cpu(), average='binary')
    recall = recall_score(test_labels.cpu(), test_preds.cpu(), average='binary')
    f1 = f1_score(test_labels.cpu(), test_preds.cpu(), average='binary')
    conf_matrix = confusion_matrix(test_labels.cpu(), test_preds.cpu())

    # Class-specific metrics for bots (class 1)
    bot_precision = precision_score(test_labels.cpu(), test_preds.cpu(), pos_label=1)
    bot_recall = recall_score(test_labels.cpu(), test_preds.cpu(), pos_label=1)
    bot_f1 = f1_score(test_labels.cpu(), test_preds.cpu(), pos_label=1)

    # Print results
    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClass-Specific Metrics for Bots (Class 1):")
    print(f"Bot Precision: {bot_precision:.4f}")
    print(f"Bot Recall: {bot_recall:.4f}")
    print(f"Bot F1-Score: {bot_f1:.4f}")

if __name__ == "__main__":
    model_path = "best_rgcn_model.pt"  # Path to the trained model
    graph_path = "graph_data.pt"  # Path to the graph data
    test_mask_path = "test_train_split/test_mask.pt"  # Path to the test mask

    # Evaluate the model with a threshold of 0.5
    evaluate_model(model_path, graph_path, test_mask_path, threshold=0.5)