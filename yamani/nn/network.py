# Author Muhammed El-Yamani

import torch.nn as nn
import torch.nn.functional as F

# Build the network
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_p = 0.2):
        super().__init__()
        # Build the network
        if len(hidden_sizes) > 0:
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
            self.layers.extend([nn.Linear(n1, n2) for n1, n2 in zip(hidden_sizes, hidden_sizes[1:])])
            self.layers.extend([nn.Linear(hidden_sizes[-1], output_size)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])
        # Build dropout
        self.drop_out = nn.Dropout(dropout_p)
    def forward(self, x):
        # iterate each layer
        for i, each in enumerate(self.layers):
            if i != len(self.layers) - 1:
                # get output of layer i
                x = each(x)
                # get acctivation relu
                x = F.relu(x)
                # make drop_out with p
                x = self.drop_out(x)
                # print('dropout')
            else:
                # last layer = output layer
                x = each(x)
                x = F.log_softmax(x, dim=1)
        return x