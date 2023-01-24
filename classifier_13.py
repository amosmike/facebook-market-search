from turtle import forward
import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import os
import time

class Identity(torch.nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TransferLearning(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = resnet50(weights=ResNet50_Weights.DEFAULT)
        # print(dir(self.layers))
        # print(self.layers.modules)

        for param in self.layers.parameters():
            param.grad_required = False

        # self.layers.avgpool = Identity()
        linear_layers = torch.nn.Sequential(
            # torch.nn.Linear(2048, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 13),
            torch.nn.Linear(2048, 13)
        )
        self.layers.fc = linear_layers

    def forward(self, X):
        return self.layers(X)

if __name__ == "__main__":
    model = TransferLearning()