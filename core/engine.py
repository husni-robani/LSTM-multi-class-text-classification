"""
This file provide several function that helps training process
"""

from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def loop_fn(mode, dataset, dataloader, model, criterion, optimizer):
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = 0

    for feature, target in tqdm(dataloader, desc=mode):
        feature = feature.squeeze()
        output = model(feature)
        loss = criterion(output, target)

        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        cost += loss.item() * feature.shape[0]
    
    cost = cost / len(dataset)
    return cost

def model_evaluation_with_confusion_matrix(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.squeeze()
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    accuracy = np.trace(cm) / np.sum(cm) * 100
    print(f'Accuracy on test data: {accuracy:.2f}%')

    # Compute precision, recall, and F1-score
    report = classification_report(all_labels, all_predictions, target_names=[f"Class {i}" for i in range(5)])
    print("Classification Report:")
    print(report)
