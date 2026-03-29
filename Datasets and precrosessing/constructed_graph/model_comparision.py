# model_comparison_results.py

results = {
    "GCN": {
        "Train Accuracy": 0.7831,
        "Test Accuracy": 0.7921,
        "Bot Recall": 0.6870
    },

    "GAT": {
        "Train Accuracy": 0.7954,
        "Test Accuracy": 0.8167,
        "Bot Recall": 0.8453
    },

    "GraphSAGE": {
        "Train Accuracy": 0.8814,
        "Test Accuracy": 0.8716,
        "Bot Recall": 0.8885
    },

    "RGCN": {
        "Train Accuracy": 0.8950,
        "Test Accuracy": 0.8823,
        "Bot Recall": 0.9029
    }
}

print("\n=== MODEL COMPARISON ===\n")
print(f"{'MODEL':<12} {'TRAIN ACC':<12} {'TEST ACC':<12} {'BOT RECALL'}")
print("-" * 55)

for model, metrics in results.items():
    print(f"{model:<12} {metrics['Train Accuracy']:<12} {metrics['Test Accuracy']:<12} {metrics['Bot Recall']}")

# Save CSV
import csv
with open("final_model_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Train Accuracy", "Test Accuracy", "Bot Recall"])
    for model, m in results.items():
        writer.writerow([
            model,
            m["Train Accuracy"],
            m["Test Accuracy"],
            m["Bot Recall"]
        ])

print("\nSaved results to final_model_results.csv")
