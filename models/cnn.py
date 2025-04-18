from skorch import NeuralNetRegressor
import torch
from torch import nn

class Conv1DModel(nn.Module):
    def __init__(self, input_dim):
        super(Conv1DModel, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dense layer before the output
        self.fc1 = nn.Linear(64 * ((input_dim - 2) // 2 - 2), 50)  # Calculate based on input size
        self.fc2 = nn.Linear(50, 1)  # Output layer

    def forward(self, x):
        # Forward pass through the model
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = NeuralNetRegressor(
    module=Conv1DModel,
    max_epochs=20,
    lr=0.01,
    iterator_train__shuffle=True
)