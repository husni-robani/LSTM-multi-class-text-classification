import pandas as pd
from core.model import LSTMClassifier
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from core.engine import model_evaluation_with_confusion_matrix
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from core.data_setup import DataSetup
from core.text_dataset import TextDataset

class EvaluatePretrainedModel():
    def __init__(self, model, logs):
        self.__model = model
        self.__logs = logs
        
    def generate_confusion_matrix_model(self):
        try:
            df = pd.read_csv('data/abstract_dataset.csv')
            w2v_model = Word2Vec.load("core/w2v_model/w2v_model.w2v")
        except FileNotFoundError as e:
            print("File Not Found Error")
        # split data
        X_train, X_test, y_train, y_test= train_test_split(df['abstract'].values, df['study_program'].values, stratify=df['study_program'], shuffle=True, random_state=42)
        
        # insert to datasetup
        datasetup = DataSetup(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, w2v_model=w2v_model)

        try:
            print("=======================PROCESSING DATA TEST START=======================")
            X_test, y_test = datasetup.processing_data_test()
        except Exception as e:
            print(f"Processing Data Error")
        else:
            print('\n')
            print("=======================PROCESSING DONE=======================")

        # make dataset
        try:
            test_set = TextDataset(texts=X_test, labels=y_test)
        except:
            print("TextDataset Error")

        # make dataloader
        test_loader = DataLoader(test_set, batch_size=64)

        # generate confusion matrix
        model_evaluation_with_confusion_matrix(model=self.__model, test_loader=test_loader)
    
    def generate_plot_cost(self):
        train_cost = self.__logs['train_cost']
        test_cost = self.__logs['test_cost']
        epochs = self.__logs['plot_tick']

        print(f"TOTAL EPOCH: {len(epochs)}")
        print(f"BEST TEST COST: {self.__logs['best_cost']}")
        print(f"\nTEST COST: {test_cost}")
        print(f"\nTRAIN COST: {train_cost}")

        # Plot train and test log loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_cost, label='Training Log Loss')
        plt.plot(epochs, test_cost, label='Testing Log Loss')

        # Set the x-ticks to be at every epoch
        plt.xticks(range(min(epochs), max(epochs) + 1, 1))

        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Training and Testing Log Loss During Training')
        plt.legend()
        plt.grid(True)
        plt.show()


        
config = {
    "input_size": 128,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 5,
    "dropout": 0.2,
    "learning_rate": 0.0001
}
# define model
model = LSTMClassifier(input_size=config["input_size"], hidden_size=config["hidden_size"], num_layers=config['num_layers'], num_classes=config["num_classes"], dropout=config["dropout"])
model.load_state_dict(torch.load('log/model_4/lstm_model.pt'))

# define logs
logs = torch.load('log/model_4/logs.pth')

# define class
model_evaluation_obj = EvaluatePretrainedModel(model= model, logs=logs)

# generate confusion matrix
# model_evaluation_obj.generate_confusion_matrix_model()

# generate plot cost
model_evaluation_obj.generate_plot_cost()