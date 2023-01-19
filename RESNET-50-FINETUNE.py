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
from torch.utils.data import random_split
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

def train(model, train_loader, val_loader, test_loader, epochs=20):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.0001)
    writer = SummaryWriter()
    batch_idx = 0
    loss_total = 0
    hist_accuracy = []

    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        for batch in pbar:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels) # Loss model changes label size 
            loss.backward()
            loss_total += loss.item()

            # loss, loss_total, hist_accuracy = calc_accuracy(prediction, labels, loss_total, hist_accuracy)

            accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
            hist_accuracy.append(accuracy)

            optimiser.step()
            optimiser.zero_grad() # gradient value reset
            batch_idx += 1

            pbar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {round(np.mean(hist_accuracy), 2)}, Loss = {round(loss.item(), 2)}, Average Loss = {round(loss_total/batch_idx, 2)}" ) # Loss = {round(loss.item(), 2)},
            writer.add_scalar('Loss = ', loss.item(), batch_idx)
            # writer.add_scalar(f'Average Loss = {round(loss_total/batch_idx, 2)}', batch_idx)
            # writer.add_scalar(f'Accuracy = {round(np.mean(hist_accuracy), 2)}', batch_idx)
        
        torch.save(model.state_dict(), f'model_evaluation/weights/ResNetCNN_{date_time}_epoch_{epoch+1}_acc_{round(np.mean(hist_accuracy), 2)}.pt')      
        
        # evaluate the validation set performance
        print('Evaluating on valiudation set')
        val_loss, val_acc = evaluate(model, val_loader)
        writer.add_scalar("Loss/Val", round(val_loss, 2), batch_idx)
        writer.add_scalar("Accuracy/Val", round(val_acc, 2), batch_idx)

    print('Evaluating on test set')
    test_loss = evaluate(model, test_loader)
    # writer.add_scalar("Loss/Test", test_loss, batch_idx)
    model.test_loss = test_loss
    final_model_fn='test_final_model.pt'
    torch.save(model.state_dict(), final_model_fn)
    return model   # return trained model

def evaluate(model, dataloader):
    losses = []
    correct = 0
    n_examples = 0
    for batch in dataloader:
        features, labels = batch
        prediction = model(features)
        loss = F.cross_entropy(prediction, labels)
        losses.append(loss.detach())
        correct += torch.sum(torch.argmax(prediction, dim=1) == labels)
        n_examples += len(labels)
    avg_loss = np.mean(losses)
    accuracy = correct / n_examples
    print("Loss:", avg_loss, "Accuracy:", accuracy.detach().numpy())
    return avg_loss, accuracy

def split_dataset(dataset):
    train_set_len = round(0.7*len(dataset))
    val_set_len = round(0.15*len(dataset))
    test_set_len = len(dataset) - val_set_len - train_set_len
    split_lengths = [train_set_len, val_set_len, test_set_len]
    train_set, val_set, test_set = random_split(dataset, split_lengths)
    return train_set,val_set,test_set

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
    train_set,val_set,test_set = split_dataset(dataset)

    batch_size=16
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # TRAIN MODEL
    print("Training Model...")
    train(model, train_loader, val_loader, test_loader)
    torch.save(model.state_dict(), f'model_evaluation/weights/ResNetCNN_{date_time}.pt')
    print("Training Successfull")


    #  USE PRE TRAINED MODEL
    # print("Loading Existing Model...")
    # state_dict = torch.load('model_evaluation/weights/ResNetCNN_2023_01_14.pt')
    # model.load_state_dict(state_dict)
    # print("Re-training...")
    # train(model, train_loader, 1)
    # torch.save(model.state_dict(), 'model_evaluation/weights/ResNetCNN_2023_01_14.pt')
    # print("Re-training Successfull")