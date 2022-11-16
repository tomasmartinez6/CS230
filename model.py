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
        self.activation_fn = nn.ReLU()
        ##############################################################################################
        
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        # put sigmoid?
        return x