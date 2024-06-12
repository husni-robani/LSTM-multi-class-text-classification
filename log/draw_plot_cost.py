import torch
import matplotlib.pyplot as plt

checkpoint = torch.load('log/model_4/logs.pth')

# Assume log loss values for training and testing are stored in the checkpoint
train_cost = checkpoint['train_cost']
test_cost = checkpoint['test_cost']
epochs = checkpoint['plot_tick']
print(f"Train cost: {train_cost}")
print(f"Test cost: {test_cost}")
print(f"Best cost: {checkpoint['best_cost']}")

# # Plot train and test log loss
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_cost, label='Training Log Loss')
# plt.plot(epochs, test_cost, label='Testing Log Loss')

# # Set the x-ticks to be at every epoch
# plt.xticks(range(min(epochs), max(epochs) + 1, 1))

# plt.xlabel('Epoch')
# plt.ylabel('Log Loss')
# plt.title('Training and Testing Log Loss During Training')
# plt.legend()
# plt.grid(True)
# plt.show()