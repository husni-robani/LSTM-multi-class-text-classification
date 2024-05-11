from core.data_setup import DataSetup
from core.text_dataset import TextDataset
from core import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

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
    print("=============PREPROCESSING START=============")
    X_train, X_test, y_train, y_test = datasetup.processing_data()
except Exception as e:
    raise "Error on preprocessing"
else:
    print('\n')
    print("=============PREPROCESSING DONE=============")

# Dataset


# DataLoader

# training loop
