import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from sklearn.datasets import make_classification
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from NN_dataset import ProductImageCategoryDataset
# from amy_image_loader import ProductImageCategoryDataset
from image_loader import ProductImageCategoryDataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class image_classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        return self.layers(X)

def train(model, epochs=10):

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
            loss = F.cross_entropy(prediction, labels) # Loss model changes label size 
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
            pbar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {round(torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels), 2)}, Total_acc = {round(np.mean(hist_accuracy), 2)}, Losses = {round(loss.item(), 2)}" )
        print('Total loss:', loss_total/batch_idx)

if __name__ == '__main__':
    # If model.pt does not exist;
    dataset = ProductImageCategoryDataset()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=8)
    model = image_classifier()
    train(model)
    torch.save(model.state_dict(), 'model-02.pt')   