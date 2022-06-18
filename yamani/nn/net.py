# Author Muhammed El-Yamani

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class NET(nn.Module):
    """Custom convolutional neural network for cloth """
    # Optimized for image size (128, 128)

    def __init__(self, input_channels=3, n_classes=46, hidden_sizes=[], dropout_p=0.2):
        super(NET, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.n_blocks = 5
        self.n_filters = 32
        self.hidden_sizes=hidden_sizes
        self.size_feature_discreptor = 512
        self.dropout_p = dropout_p

        # Build conv
        self._build_conv()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        # Build fc
        self._build_fc()
        # Build dropout
        self.drop_out = nn.Dropout(self.dropout_p)

    def forward(self, x):
        for each in self.conv:
            # get output of layer i
            x = each(x)
        # this step to make sure the input size the fc unified
        x = self.avgpool(x)
        # flatten the input (c,w,h)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # iterate each layer
        for i, each in enumerate(self.fc):
            if i != len(self.fc) - 1:
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

    def _build_conv(self):

        self.conv = nn.ModuleList()
        for i in range(self.n_blocks):
            if i == 0:
                input_block = self.input_channels
            else:
                input_block = self.n_filters
            output_block = self.n_filters

            self.conv.append(BLOCK(input_block, output_block))

    def _build_fc(self):
        if len(self.hidden_sizes) > 0:
            self.fc = nn.ModuleList(
                [nn.Linear(self.size_feature_discreptor, self.hidden_sizes[0])])
            self.fc.extend([nn.Linear(n1, n2)
                            for n1, n2 in zip(self.hidden_sizes, self.hidden_sizes[1:])])
            self.fc.extend([nn.Linear(self.hidden_sizes[-1], self.n_classes)])
        else:
            self.fc = nn.ModuleList(
                [nn.Linear(self.size_feature_discreptor, self.n_classes)])

class BLOCK(nn.Module):
    """Repeated block in NET """

    def __init__(self, input_channels, output_channels):
        super(BLOCK, self).__init__()

        self.input_channels = input_channels
        self.n_filters = output_channels

        self.block = nn.Sequential(
            nn.Conv2d(self.input_channels, output_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channels),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        )

    def forward(self, x):
        x = self.block(x)
        return x
