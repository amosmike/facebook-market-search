import torch 
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
# import torch as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
from torch.utils.tensorboard import SummaryWriter
from image_loader import ProductImageCategoryDataset


class ResNetCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 1)

    def forward(self, X):
        return torch.sigmoid(self.resnet50(X))

# def train(model, dataloader, epochs=10):

#     optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
#     writer = SummaryWriter()

#     for epoch in range(epochs):
#         loss_total = 0
#         batch_idx = 0
#         for batch in dataloader:
#             features, labels = batch
#             prediction = model(features)
#             labels = labels.unsqueeze(1)
#             labels = labels.float()

#             loss = F.binary_cross_entropy(prediction, labels) # Loss model changes label size 
#             loss.backward()
#             optimiser.step() 
#             optimiser.zero_grad()
#             print('loss:', loss.item())
#             # loss_total += loss.item()
#             # writer.add_scalar('Loss', loss.item(), batch_idx)
#             # batch_idx += 1
#         # print('Total loss:', loss_total/batch_idx)

def train(model, train_loader, epochs=1):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    batch_idx = 0
    loss_total = 0

    for epoch in range(epochs):
        hist_accuracy = []
        pbar = tqdm(train_loader)
        for batch in pbar:
            features, labels = batch
            prediction = model(features)
            labels = labels.unsqueeze(1)
            labels = labels.float()

            loss = F.binary_cross_entropy(prediction, labels) # Loss model changes label size 
            loss_total += loss.item()
            loss.backward()
            # print('loss:', loss.item())
            optimiser.step() 
            optimiser.zero_grad() # gradient value reset 
            writer.add_scalar('Loss', loss.item(), batch_idx)
            accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
            hist_accuracy.append(accuracy)
            # pbar.set_description(f"Loss: {loss.item()}. Epoch: {epoch}/{epochs}")
            batch_idx += 1
            pbar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {round(np.mean(hist_accuracy), 2)}, Total loss = {round(loss_total/batch_idx)}" ) # Loss = {round(loss.item(), 2)}, 

if __name__ == '__main__':
    classifier = ResNetCNN()
    dataset = ProductImageCategoryDataset()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=8)
    train(classifier, train_loader, 1)
    torch.save(classifier.state_dict(), 'ResNetCNN-1-epoch.pt')   

# RuntimeError: expected scalar type Byte but found Float