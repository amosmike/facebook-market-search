#%%
import torch 
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from sklearn.datasets import make_classification
from torch.utils.tensorboard import SummaryWriter
from image_loader import ProductImageCategoryDataset

class ResNetCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = resnet50(weights=ResNet50_Weights.DEFAULT)
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


    def forward(self, X):
        return self.layers(X)

def train(model, train_loader, epochs=5):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    batch_idx = 0
    loss_total = 0

    for epoch in range(epochs):
        hist_accuracy = []
        pbar = tqdm(train_loader)
        for batch in pbar:
        # for batch in train_loader:
            features, labels = batch
            prediction = model(features)

            loss = F.cross_entropy(prediction, labels) # Loss model changes label size 
            loss.backward()
            # print(loss.item())
            loss_total += loss.item()

            optimiser.step()
            optimiser.zero_grad() # gradient value reset

            writer.add_scalar('Loss', loss.item(), batch_idx)
            accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
            hist_accuracy.append(accuracy)
            batch_idx += 1
            pbar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {round(np.mean(hist_accuracy), 2)}, Total loss = {round(loss_total/batch_idx)}" ) # Loss = {round(loss.item(), 2)},
        
        torch.save(model.state_dict(), f'model_evaluation/weights/ResNetCNN_{date_time}_epoch_{epoch+1}.pt')      

if __name__ == '__main__':
    date_time = datetime.today().strftime('%Y_%m_%d')
    model = ResNetCNN()
    
    image_size=64
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop((image_size), pad_if_needed=True),
        transforms.ToTensor(),
        ])    
    
    dataset = ProductImageCategoryDataset(transform=transform)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=16)

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