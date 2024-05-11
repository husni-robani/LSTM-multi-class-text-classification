from torch.utils.data import Dataset
import re
import numpy as np
import pandas as pd

class TextDataset(Dataset):
    """This is custom dataset for abstract text"""
    def __init__(self, texts, labels):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]