import torch

def inspect_graph_data(file_path):
    try:
        # Load the graph data
        graph_data = torch.load(file_path)

        # Check if the graph data contains node labels
        if hasattr(graph_data, 'y'):
            labels = graph_data.y.numpy()
            unique, counts = torch.unique(graph_data.y, return_counts=True)
            label_distribution = dict(zip(unique.tolist(), counts.tolist()))

            print("Label Distribution:")
            for label, count in label_distribution.items():
                print(f"Label {label}: {count} nodes")
        else:
            print("The graph data does not contain node labels.")

        # Additional graph information
        print("\nGraph Information:")
        print(f"Number of nodes: {graph_data.num_nodes}")
        print(f"Number of edges: {graph_data.num_edges}")
        print(f"Edge index shape: {graph_data.edge_index.shape}")
        print(f"Feature matrix shape: {graph_data.x.shape}")

    except Exception as e:
        print(f"Error loading graph data: {e}")

if __name__ == "__main__":
    file_path = "graph_data.pt"  # Path to the graph data file
    inspect_graph_data(file_path)