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

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=(
                3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),

            nn.Conv2d(self.input_channels, 32, kernel_size=(
                3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),

            nn.Conv2d(self.input_channels, 32, kernel_size=(
                3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),

            nn.Conv2d(self.input_channels, 32, kernel_size=(
                3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),

            nn.Conv2d(self.input_channels, 32, kernel_size=(
                3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),

        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

        size_feature_discreptor = 256

        self.fc = nn.Linear(in_features=size_feature_discreptor,
                            out_features=self.n_classes)

        if len(hidden_sizes) > 0:
            self.fc = nn.ModuleList(
                [nn.Linear(size_feature_discreptor, hidden_sizes[0])])
            self.fc.extend([nn.Linear(n1, n2)
                            for n1, n2 in zip(hidden_sizes, hidden_sizes[1:])])
            self.fc.extend([nn.Linear(hidden_sizes[-1], self.n_classes)])
        else:
            self.fc = nn.ModuleList(
                [nn.Linear(size_feature_discreptor, self.n_classes)])
        # Build dropout
        self.drop_out = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
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

