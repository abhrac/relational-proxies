import torch
from torch import nn


class RelationNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, features_1, features_2):
        pair = torch.cat([features_1, features_2], dim=1)
        return self.layers(pair)
