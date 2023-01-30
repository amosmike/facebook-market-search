#%%
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50
from torchvision import models
from torchvision import transforms
import torch

class TransferLearning(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = resnet50(weights=ResNet50_Weights.DEFAULT)

        # print(dir(self.layers))
        # print(self.layers.modules)
        # print(self.layers.parameters())
        print(self.layers)


        for i, param in enumerate(self.layers.parameters()): # unfreeze last two layers (4.1 and 4.2) of resnet50
            if i > 141 and i < 160:
                param.requires_grad=False
            else:
                param.requires_grad=True

        linear_layers = torch.nn.Sequential(
            torch.nn.Linear(2048, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 13)
            )

        self.layers.fc = linear_layers
        self.image_size = 128
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomCrop((self.image_size), pad_if_needed=True),
            transforms.ToTensor(),
            ])

    def forward(self, X):
        return self.layers(X)

if __name__ == "__main__":
    model = TransferLearning()