from core.data_setup import DataSetup
from core.text_dataset import TextDataset
from core.model import LSTMClassifier
from core import utils
from core import engine
from jcopdl.callback import Callback
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.utils.data import DataLoader

try: 
    df = pd.read_csv("data/abstract_dataset.csv")
    w2v_model = Word2Vec.load("core/w2v_model/w2v_model.w2v")
except FileNotFoundError as e:
    raise f"FileError: {e}"

# train test split
X_train, X_test, y_train, y_test = train_test_split(df['abstract'].values, df['study_program'].values, stratify=df['study_program'], shuffle=True, random_state=42)

# Preprocessing Data
datasetup = DataSetup(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, w2v_model=w2v_model, is_oversampling=True)
try:
    print("==========================PREPROCESSING START==========================")
    X_train, X_test, y_train, y_test = datasetup.processing_data()
except Exception as e:
    raise f"Error on preprocessing \n {e}"
else:
    print('\n')
    print("==========================PREPROCESSING DONE==========================")

# Dataset
try:
    train_set = TextDataset(texts=X_train, labels=y_train)
    test_set = TextDataset(texts=X_test, labels=y_test)
except Exception as e:
    raise f"Error on creating dataset\n{e}"

# DataLoader
try:
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=64)
except Exception as e:
    raise f"Error on creating dataloader\n{e}"

train_iter, test_iter = iter(train_loader), iter(test_loader)
train_batch, test_batch = next(train_iter), next(test_iter)
train_texts, train_labels = train_batch
test_texts, test_labales = test_batch

print(f"Shape train texts : {train_texts.shape} | Shape train labels: {train_labels.shape}")
print(f"Shape test texts : {test_texts.shape} | Shape test labels: {test_labales.shape}")

# training loop

config = {
    "input_size": 128,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 5,
    "dropout": 0.2,
    "learning_rate": 0.0001
}

# defining m-c-o-c
model = LSTMClassifier(input_size=config["input_size"], hidden_size=config["hidden_size"], num_layers=config['num_layers'], num_classes=config["num_classes"], dropout=config["dropout"])
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss()
callback = Callback(model=model, outdir='log')


print('\n')
print("==========================TRAINING START==========================")

while True:
    # data training
    train_cost = engine.loop_fn("train", train_set, train_loader, model, criterion, optimizer)

    # data test
    with torch.no_grad():
        test_cost = engine.loop_fn("test", test_set, test_loader, model, criterion, optimizer)
    
    # Callback
        # logging
    callback.log(train_cost=train_cost, test_cost=test_cost)

        # checkpoint
    callback.save_checkpoint()

        # Runtime Plotting
    callback.cost_runtime_plotting()

        # Early Stopping
    stoper = callback.early_stopping(model=model, monitor="test_cost")
    if stoper:
        callback.plot_cost()
        break

print('\n')
print("==========================TRAINING END==========================")

# Save Models
torch.save(model.state_dict(), "log/lstm_model.pt")

# Confusion Matrix
engine.model_evaluation_with_confusion_matrix(model=model, test_loader=test_loader)