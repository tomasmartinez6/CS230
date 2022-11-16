import torch.nn as nn
import torch.nn.functional as F


class TwoLayerModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        # Create two layer network, where each layer has same number of neurons as LogisticRegression
        # model used in Baseline.
        # Since LogReg uses input_dims neurons, we can make hidden_dims = input_dims
        ##############################################################################################
        # CONFIGURATIONS
        hidden_dims = input_dims
        self.relu1 = nn.ReLU()
        self.relu2 = nn.Tanh()
        
        ##############################################################################################
        
        self.fc1 = nn.Linear(input_dims, hidden_dims, bias=True)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims, bias=True)
        self.output_linear= nn.Linear(hidden_dims, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output_linear(x)
        x = self.sigmoid(x)
        # put sigmoid?
        return x