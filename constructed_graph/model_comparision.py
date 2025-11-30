import pandas as pd

data = {
    "Model": ["GraphSAGE","R-GCN","GCN","GAT"],
    "Best Val Acc": [0.89,0.89,0.80,0.86],
    "Test Acc": [0.89,0.91,0.83,0.88],
    "Model File": ["best_sage_model.pt","best_rgcn_model.pt","best_gcn.pt","best_gat.pt"]
}

df = pd.DataFrame(data)
df.to_csv("model_comparison.csv", index=False)
print(df)
print("\nSaved → model_comparison.csv")
