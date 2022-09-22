import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from sklearn.datasets import make_classification
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from image_loader import ProductImageCategoryDataset

class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 1)

    def forward(self, X):
        return F.sigmoid(self.resnet50(X))

def train(model, dataloader, epochs=10):

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    for epoch in range(epochs):
        loss_total = 0
        batch_idx = 0
        for batch in dataloader:
            features, labels = batch
            prediction = model(features)
            labels = labels.unsqueeze(1)
            labels = labels.float()

            loss = F.binary_cross_entropy(prediction, labels) # Loss model changes label size 
            loss.backward()
            optimiser.step() 
            optimiser.zero_grad()
            print('loss:', loss.item())
            # loss_total += loss.item()
            # writer.add_scalar('Loss', loss.item(), batch_idx)
            # batch_idx += 1
        # print('Total loss:', loss_total/batch_idx)

if __name__ == '__main__':
    classifier = CNN()
    dataset = ProductImageCategoryDataset()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=8)
    train(classifier, data_loader, 10)

# RuntimeError: expected scalar type Byte but found Float