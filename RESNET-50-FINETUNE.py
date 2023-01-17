import torch 
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
# import torch as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from image_loader import ProductImageCategoryDataset


class ResNetCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.resnet50 = torch.hub.load(
        #     'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.layers = resnet50(weights=ResNet50_Weights)
        for param in self.layers.parameters():
            param.grad_required = False
        
        linear_layers = torch.nn.Sequential(
            # torch.nn.Linear(2048, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 13),
            torch.nn.Linear(2048, 13)
        )
        self.layers.fc = linear_layers

        # print(self.layers.parameters)
        # torch.nn.Linear(1, 13) # IS THIS RIGHT?
        
        # torch.nn.Linear(193600, 1000) # last Linear number is number of classes from dataset


        # self.resnet50.avgpool = torch.nn.AdaptiveAvgPool2d(13)
        # self.resnet50.fc = torch.nn.Linear(346112, 13)


    def forward(self, X):
        return self.layers(X)

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
        
        torch.save(model.state_dict(), f'model_evaluation/weights/ResNetCNN_2023_01_14_epoch_{epoch+1}.pt')      

if __name__ == '__main__':
    # date_time = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    model = ResNetCNN()
    dataset = ProductImageCategoryDataset()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=8)

    # TRAIN MODEL
    print("Training Model...")
    train(model, train_loader)
    torch.save(model.state_dict(), 'model_evaluation/weights/ResNetCNN_2023_01_14.pt')
    print("Training Successfull")


    #  USE PRE TRAINED MODEL
    # print("Loading Existing Model...")
    # state_dict = torch.load('model_evaluation/weights/ResNetCNN_2023_01_14.pt')
    # model.load_state_dict(state_dict)
    # print("Re-training...")
    # train(model, train_loader, 1)
    # torch.save(model.state_dict(), 'model_evaluation/weights/ResNetCNN_2023_01_14.pt')
    # print("Re-training Successfull")