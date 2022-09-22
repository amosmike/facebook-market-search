import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from sklearn.datasets import make_classification
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from image_loader import ProductImageCategoryDataset


def train(model, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter()

    for epoch in range(epochs):
        loss_total = 0
        batch_idx = 0
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels) # Loss model changes label size 
            loss_total += loss.item()
            loss.backward()
            print('loss:', loss.item())
            optimiser.step() 
            optimiser.zero_grad()
            writer.add_scalar('Loss', loss.item(), batch_idx)
            batch_idx += 1
        # print('Total loss:', loss_total/batch_idx)

            
class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.linear_layer = torch.nn.Linear(64, 16) #(No. features, no. layers) # ERROR
        # self.linear_layer2 = torch.nn.Linear(16, 1)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3), # (input channels = LGB, output channels = how many features (can change), no. kernals) | image shape (3, 64, 64)
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 8), # (input = output from first conv2d, features wanting to capture from feature, kernal size)
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(193600, 13) # last Linear number is number of classes from dataset
            # torch.nn.Softmax(),
        )

    def forward(self, X):
        # print(X.size())
        # X = self.linear_layer(X)
        # X = F.relu(X)
        # X = self.linear_layer2(X)
        # return X
        return self.layers(X)

if __name__ == '__main__':
    dataset = ProductImageCategoryDataset()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=8)
    model = CNN()
    # train(model, train_loader)
    train(model)

# RuntimeError: expected scalar type Byte but found Float