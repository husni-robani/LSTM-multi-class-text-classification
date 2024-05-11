import torch
from torch import nn

class LSTMClassifier(nn.Module):
    """this is LSTM custom model class"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size//4, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))

        # Flatten the output from LSTM layer if needed
        out = out[:, -1, :]  # Get the last time step output
        # You may need to flatten the output if needed, depending on the shape

        out = self.fc(out)
        return out
    
