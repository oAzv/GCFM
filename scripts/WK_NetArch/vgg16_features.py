#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import nn


class EncoderCNN(nn.Module):
    '''
    Vgg16 pre-training model call, network structure adaptation modification.
    '''
    def __init__(self, model):
        '''Initialize the model, deconstruct it.

        Args:

            model: Vgg16 pre-training model.
        '''
        super(EncoderCNN, self).__init__()

        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        self.Linear_layer = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 800)
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512, 2)
        )

    def forward(self, x):
        '''Forward calculation to extract the required features.

        Args:
            x: The gray image data.

        Returns:
            The SIR feature.
        '''
        x = self.feature(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        x = x.view(x.size(0), -1)

        x = self.Linear_layer(x)  # (1, 10)
        # x = x.view(x.size(0), -1)

        return x