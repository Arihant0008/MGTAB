from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Define the RGCN model architecture
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

# Load your trained model and graph data
print("Loading model and graph data...")
graph_data = torch.load('graph_data.pt', weights_only=False)

# Get number of relations
num_rel = int(graph_data.edge_type.max().item() + 1)

# Create model and load weights
model = RGCN(graph_data.x.shape[1], 256, 2, num_rel)
model.load_state_dict(torch.load('best_rgcn_model.pt'))
model.eval()
print("✅ Model loaded successfully!")

# Get feature statistics for normalization
feature_mean = graph_data.x.mean(dim=0)
feature_std = graph_data.x.std(dim=0)

def normalize_feature(value, min_val, max_val):
    """Normalize a value to [0, 1] range"""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def create_feature_vector(user_input):
    """
    Convert 20 user features to 788-dimensional vector with proper normalization
    """
    # Approximate normalization ranges based on Twitter data
    # These should match the training data preprocessing
    
    # Binary features (0 or 1)
    binary_features = {
        'profile_use_background_image': user_input.get('profile_use_background_image', 0),
        'default_profile': user_input.get('default_profile', 0),
        'verified': user_input.get('verified', 0),
        'default_profile_image': user_input.get('default_profile_image', 0),
        'geo_enabled': user_input.get('geo_enabled', 0),
        'default_profile_background_color': user_input.get('default_profile_background_color', 0),
        'default_profile_sidebar_fill_color': user_input.get('default_profile_sidebar_fill_color', 0),
        'default_profile_sidebar_border_color': user_input.get('default_profile_sidebar_border_color', 0),
        'has_URL': user_input.get('has_URL', 0),
        'profile_background_image_URL': user_input.get('profile_background_image_URL', 0),
    }
    
    # Numeric features - normalize to [0, 1]
    followers = normalize_feature(user_input.get('followers_count', 0), 0, 100000)
    friends = normalize_feature(user_input.get('friends_count', 0), 0, 100000)
    statuses = normalize_feature(user_input.get('statuses_count', 0), 0, 100000)
    listed = normalize_feature(user_input.get('listed_count', 0), 0, 1000)
    favourites = normalize_feature(user_input.get('favourites_count', 0), 0, 100000)
    created_at = normalize_feature(user_input.get('created_at', 0), 0, 5000)  # days
    screen_name_len = normalize_feature(user_input.get('screen_name_length', 0), 1, 15)
    name_len = normalize_feature(user_input.get('name_length', 0), 1, 50)
    desc_len = normalize_feature(user_input.get('description_length', 0), 0, 160)
    ff_ratio = min(user_input.get('followers_friends_ratios', 0) / 10.0, 1.0)  # Cap at 10
    
    # Order of features as they appear in the model
    features_20 = torch.tensor([
        binary_features['profile_use_background_image'],
        binary_features['default_profile'],
        binary_features['verified'],
        followers,
        binary_features['default_profile_image'],
        listed,
        statuses,
        friends,
        binary_features['geo_enabled'],
        favourites,
        created_at,
        screen_name_len,
        name_len,
        desc_len,
        ff_ratio,
        binary_features['default_profile_background_color'],
        binary_features['default_profile_sidebar_fill_color'],
        binary_features['default_profile_sidebar_border_color'],
        binary_features['has_URL'],
        binary_features['profile_background_image_URL']
    ], dtype=torch.float32)
    
    print(f"Normalized features (first 20): {features_20.tolist()}")
    
    # Find the most similar node based on first 20 features
    similarities = F.cosine_similarity(
        features_20.unsqueeze(0).expand(graph_data.x.shape[0], -1),
        graph_data.x[:, :20],
        dim=1
    )
    most_similar_idx = similarities.argmax().item()
    most_similar_label = graph_data.y[most_similar_idx].item()
    
    print(f"Most similar node: {most_similar_idx} (Label: {'BOT' if most_similar_label == 1 else 'HUMAN'}), Similarity: {similarities[most_similar_idx]:.4f}")
    
    # Use the most similar node's features for dimensions 20-788
    # This preserves the tweet embedding structure
    features_788 = graph_data.x[most_similar_idx].clone()
    features_788[:20] = features_20  # Override with user input
    
    # Add logic to extract relationship-based features
    # Explicit relationships
    follower_count = user_input.get('follower_count', 0)
    friend_count = user_input.get('friend_count', 0)
    mention_count = user_input.get('mention_count', 0)
    reply_count = user_input.get('reply_count', 0)
    quote_count = user_input.get('quote_count', 0)

    # Implicit relationships
    shared_hashtags = user_input.get('shared_hashtags', 0)
    shared_urls = user_input.get('shared_urls', 0)

    # Normalize relationship features
    normalized_relationship_features = torch.tensor([
        normalize_feature(follower_count, 0, 100000),
        normalize_feature(friend_count, 0, 100000),
        normalize_feature(mention_count, 0, 1000),
        normalize_feature(reply_count, 0, 1000),
        normalize_feature(quote_count, 0, 1000),
        normalize_feature(shared_hashtags, 0, 100),
        normalize_feature(shared_urls, 0, 100)
    ], dtype=torch.float32)

    # Combine attributes and relationships
    features_combined = torch.cat([features_20, normalized_relationship_features], dim=0)

    return features_788.unsqueeze(0)  # Add batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        user_data = request.json
        print(f"Received data: {user_data}")
        
        # Create feature vector for new user
        new_features = create_feature_vector(user_data)
        
        # Get number of existing nodes
        num_existing_nodes = graph_data.x.shape[0]
        new_node_idx = num_existing_nodes
        
        # Add new node to existing graph features
        extended_features = torch.cat([graph_data.x, new_features], dim=0)
        
        # Create edges connecting new node to existing graph
        # Strategy: Connect to K nearest neighbors based on feature similarity
        K = 20  # Connect to 20 most similar nodes (increase for better context)
        
        # Calculate similarity between new node and existing nodes (only first 20 features)
        # Focus on the actual user features, not the padding
        similarities = F.cosine_similarity(
            new_features[:, :20].expand(num_existing_nodes, -1),
            graph_data.x[:, :20],
            dim=1
        )
        
        # Get indices of K most similar nodes
        top_k_values, top_k_indices = torch.topk(similarities, K)
        
        print(f"Top similarities: {top_k_values[:5].tolist()}")
        print(f"Connected to nodes: {top_k_indices[:5].tolist()}")
        
        # Check labels of similar nodes
        similar_labels = graph_data.y[top_k_indices].tolist()
        bot_count = sum(similar_labels)
        human_count = len(similar_labels) - bot_count
        print(f"Similar nodes - Bots: {bot_count}, Humans: {human_count}")
        
        # Create bidirectional edges between new node and K nearest neighbors
        new_edges_source = []
        new_edges_target = []
        new_edge_types = []
        
        for i, similar_idx in enumerate(top_k_indices):
            # Use different edge types based on feature similarity
            edge_type = 0 if i < K//2 else 1  # Vary edge types
            
            # Edge from new node to similar node
            new_edges_source.append(new_node_idx)
            new_edges_target.append(similar_idx.item())
            new_edge_types.append(edge_type)
            
            # Edge from similar node to new node (bidirectional)
            new_edges_source.append(similar_idx.item())
            new_edges_target.append(new_node_idx)
            new_edge_types.append(edge_type)
        
        # Add relationship-based edges
        for i, similar_idx in enumerate(top_k_indices):
            edge_type = 2 if i < K//2 else 3  # Use new edge types for relationships
            new_edges_source.append(new_node_idx)
            new_edges_target.append(similar_idx.item())
            new_edge_types.append(edge_type)

            new_edges_source.append(similar_idx.item())
            new_edges_target.append(new_node_idx)
            new_edge_types.append(edge_type)
        
        # Convert to tensors
        new_edges_source = torch.tensor(new_edges_source, dtype=torch.long)
        new_edges_target = torch.tensor(new_edges_target, dtype=torch.long)
        new_edge_types = torch.tensor(new_edge_types, dtype=torch.long)
        
        # Combine new edges with existing edges
        extended_edge_index = torch.cat([
            graph_data.edge_index,
            torch.stack([new_edges_source, new_edges_target], dim=0)
        ], dim=1)
        
        extended_edge_type = torch.cat([
            graph_data.edge_type,
            new_edge_types
        ], dim=0)
        
        print(f"New node index: {new_node_idx}")
        print(f"Connected to {K} similar nodes")
        print(f"Total nodes: {extended_features.shape[0]}")
        print(f"Total edges: {extended_edge_index.shape[1]}")
        
        # Run model inference on extended graph
        with torch.no_grad():
            out = model(extended_features, extended_edge_index, extended_edge_type)
            
            # Get prediction for the new node (last node)
            pred_logits = out[-1]
            pred_probs = F.softmax(pred_logits, dim=0)
            prediction = pred_probs.argmax().item()
            confidence = pred_probs.max().item()
        
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS:")
        print(f"Raw logits: {pred_logits.tolist()}")
        print(f"Probabilities - Human: {pred_probs[0]:.4f}, Bot: {pred_probs[1]:.4f}")
        print(f"Final Prediction: {'🤖 BOT' if prediction == 1 else '✅ HUMAN'}")
        print(f"Confidence: {confidence:.4f}")
        print(f"{'='*50}\n")
        
        # ---------- BUILD SUBGRAPH FOR VISUALIZATION ----------

        # Prepare neighbor list
        neighbors_list = []
        for sim_val, nidx in zip(top_k_values.tolist(), top_k_indices.tolist()):
            neighbors_list.append({
                'idx': int(nidx),
                'similarity': float(sim_val),
                'label': int(graph_data.y[nidx].item())
            })

        # Build subgraph node set (neighbors + new node)
        subgraph_node_indices = top_k_indices.tolist() + [new_node_idx]

        # Nodes JSON
        nodes_json = []
        for n in subgraph_node_indices:
            nodes_json.append({
                'id': int(n),
                'label': int(graph_data.y[n].item()) if n != new_node_idx else None,
                'first20': (graph_data.x[n][:20].tolist() if n < graph_data.x.shape[0] else new_features[0,:20].tolist())
            })

        # Edges JSON (only edges linking nodes inside subgraph)
        edges_json = []
        src = extended_edge_index[0].tolist()
        tgt = extended_edge_index[1].tolist()
        etype = extended_edge_type.tolist()

        for i, (s, t) in enumerate(zip(src, tgt)):
            if s in subgraph_node_indices and t in subgraph_node_indices:
                edges_json.append({
                    'source': int(s),
                    'target': int(t),
                    'type': int(etype[i])
                })

        # ---------- RETURN EVERYTHING ----------

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'human': pred_probs[0].item(),
                'bot': pred_probs[1].item()
            },

            # NEW FIELDS
            'new_node_index': int(new_node_idx),
            'neighbors': neighbors_list,
            'subgraph': {
                'nodes': nodes_json,
                'edges': edges_json
            }
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model statistics"""
    return jsonify({
        'model': 'R-GCN (Relational Graph Convolutional Network)',
        'test_accuracy': 0.91,
        'num_nodes': graph_data.x.shape[0],
        'num_edges': graph_data.edge_index.shape[1],
        'num_features': graph_data.x.shape[1],
        'edge_types': ['followers', 'friends', 'mentions', 'replies', 'quoted', 'urls', 'hashtags']
    })

@app.route('/', methods=['GET'])
def home():
    """Serve the main HTML interface"""
    return send_file('index.html')

if __name__ == '__main__':
    print("\n🚀 Starting Flask server...")
    print("📊 Model: R-GCN")
    print("✅ Test Accuracy: 85%")
    print("🌐 Server: http://localhost:5000\n")
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)