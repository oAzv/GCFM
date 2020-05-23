#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import nn

class EncoderCNN(nn.Module):
    def __init__(self, model):
        super(EncoderCNN, self).__init__()

        self.all = nn.Sequential(*list(model.children())[:-1])

        self.Linear_layer = nn.Sequential(
            nn.Linear(9216, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 800),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(800, 800)
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.all(x)
        x = x.view(x.size(0), -1)

        # x = self.classifier(x)
        # x = x.view(x.size(0), -1)

        x = self.Linear_layer(x)  # (1, 10)
        # x = x.view(x.size(0), -1)

        return x