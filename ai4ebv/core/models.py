"""Neural network models used for classification."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# externals
import torch.nn as nn

# locals
from pysegcnn.core.models import Network
from pysegcnn.core.layers import Conv1dSame


class MLP(Network):
    def __init__(self, in_channels, nclasses, state_file=None):
        super().__init__(state_file, in_channels, nclasses)

        # define the hidden layers
        self.do1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.in_channels, 500)
        self.do2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(500, 500)
        self.do3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(500, self.nclasses)
        self.do4 = nn.Dropout(0.3)

        # define activation function
        self.relu = nn.ReLU()

    # the forward propagation in the network
    def forward(self, x):
        # first layer
        x = self.relu(self.fc1(self.do1(x)))

        # second layer
        x = self.relu(self.fc2(self.do2(x)))

        # third layer
        x = self.relu(self.fc3(self.do3(x)))

        # output layer
        x = self.do4(x)

        # global average pooling over time series
        if x.ndim > 2:
            x = x.mean(axis=1, keepdims=False)

        return x


class FCN(Network):

    def __init__(self, in_channels, nclasses, state_file=None):
        super().__init__(state_file, in_channels, nclasses)

        # non-linear activation function
        self.relu = nn.ReLU()

        # define the convolutional layers
        self.conv1 = Conv1dSame(self.in_channels, 128, kernel_size=3)
        self.conv2 = Conv1dSame(128, 256, kernel_size=3)
        self.conv3 = Conv1dSame(256, 128, kernel_size=3)

        # define the batch normalization layers
        self.banm1 = nn.BatchNorm1d(128)
        self.banm2 = nn.BatchNorm1d(256)
        self.banm3 = nn.BatchNorm1d(128)

        # define the classification layer
        self.clf = Conv1dSame(128, self.nclasses, kernel_size=1)

    def forward(self, x):

        # forward pass: feature extraction
        x = self.relu(self.banm1(self.conv1(x)))
        x = self.relu(self.banm2(self.conv2(x)))
        x = self.relu(self.banm3(self.conv3(x)))

        # forward pass: classification
        x = self.clf(x)

        # forward pass: global average pooling
        x = x.mean(axis=-1, keepdims=False)

        return x


class ANN(Network):
    def __init__(self, in_channels, nclasses, state_file=None):
        super().__init__(state_file, in_channels, nclasses)

        # define the hidden layers
        self.do1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.in_channels, 256)
        self.fc2 = nn.Linear(256, self.nclasses)
        self.do2 = nn.Dropout(0.3)

        # define activation function
        self.relu = nn.ReLU()

    # the forward propagation in the network
    def forward(self, x):
        # first layer
        x = self.relu(self.fc1(self.do1(x)))

        # second layer
        x = self.fc2(self.do2(x))

        # global average pooling over time series
        if x.ndim > 2:
            x = x.mean(axis=1, keepdims=False)

        return x