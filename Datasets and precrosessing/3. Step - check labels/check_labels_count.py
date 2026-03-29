import torch

labels = torch.load("Dataset/labels_bot.pt")

num_nodes = labels.shape[0]
num_bot  = (labels == 1).sum().item()
num_real = (labels == 0).sum().item()

print("Total nodes:", num_nodes)
print("Bot (1):", num_bot)
print("Real (0):", num_real)

ratio_bot = num_bot / num_nodes
ratio_real = num_real / num_nodes

print("\nBot ratio:", ratio_bot)
print("Real ratio:", ratio_real)
