"""
this file provide several helpful utility functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(data:np.ndarray, title:str, class_names:list=[0, 1, 2, 3, 4]):
    dd = pd.Series(data).value_counts()
    study_program = np.array(class_names)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=study_program, y=dd.values, ax=ax)
    plt.title(title)
    plt.show()
