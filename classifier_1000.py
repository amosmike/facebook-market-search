import torch
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

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
        # print(self.layers.parameters())

        for param in self.layers.parameters():
            param.grad_required = False

        # self.layers.fc.grad_required = True
        # self.layers.fc.grad_required = True
        self.layers.fc.requires_grad = True
        self.layers.avgpool.requires_grad = True   

        # self.layers.avgpool.requires_grad_ = True

        # self.layers.avgpool = Identity()
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