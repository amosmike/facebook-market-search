#%%
import torch 
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from dataset import ImagesDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from classifier_1000 import TransferLearning
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    lr = 0.0001,
    epochs=20,
    optimiser = torch.optim.SGD
    ):

    ""

    writer = SummaryWriter()

    optimiser = optimiser(model.parameters(), lr=lr)
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
            optimiser.step()
            loss_total += loss.item()
            optimiser.zero_grad() # gradient value reset
            writer.add_scalar('Loss', loss.item(), batch_idx)

            accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
            hist_accuracy.append(accuracy)

            batch_idx += 1
            pbar.set_description(f"Epoch = {epoch+1}/{epochs}. Acc = {round(np.mean(hist_accuracy), 2)}, Loss = {round(loss.item(), 2)}, Average Loss = {round(loss_total/batch_idx, 2)}" ) # Loss = {round(loss.item(), 2)},
            # writer.add_scalar(f'Average Loss = {round(loss_total/batch_idx, 2)}', batch_idx)
            writer.add_scalar('Training Set Accuracy', np.mean(hist_accuracy), batch_idx)
        
        
        # evaluate the validation set performance
        print('Evaluating on valiudation set...')
        val_loss, val_acc = evaluate(model, val_loader)
        writer.add_scalar("Validation Set Loss", val_loss, batch_idx)
        writer.add_scalar("Validation Set Accuracy", val_acc, batch_idx)

        # Save weights
        # torch.save(model.state_dict(), f'model_evaluation/weights/transfer_learning_{date_time}_lr_{lr}_epoch_{epoch+1}_acc_{round(np.mean(hist_accuracy), 2)}.pt')      
        torch.save(model.state_dict(), f'final_model/image_model.pt')

    print('Evaluating on test set...')
    test_loss = evaluate(model, test_loader)
    # writer.add_scalar("Loss/Test", test_loss, batch_idx)
    model.test_loss = test_loss
    # final_model_fn='test_final_model.pt'
    # torch.save(model.state_dict(), final_model_fn)
    torch.save(model.state_dict(), f'final_model/image_model.pt') 
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

if __name__ == "__main__":
    date_time = datetime.today().strftime('%Y_%m_%d')
    model = TransferLearning()
    image_size = 128
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop((image_size), pad_if_needed=True),
        transforms.ToTensor(),
        ]) 

    dataset = ImagesDataset(transform=transform)
    train_set, val_set, test_set = split_dataset(dataset)

    batch_size = 16
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=20,
        lr=0.0001,
        optimiser=torch.optim.AdamW
    )

    torch.save(model.state_dict(), f'final_model/image_model.pt')
    print("Training Successfull")

    # USE PRE TRAINED MODEL
    # print("Loading Existing Model...")
    # state_dict = torch.load('model_evaluation/weights/ResNetCNN_2023_01_14.pt')
    # model.load_state_dict(state_dict)
    # print("Re-training...")
    # train(model, train_loader, 1)
    # torch.save(model.state_dict(), 'model_evaluation/weights/ResNetCNN_2023_01_14.pt')
    # print("Re-training Successfull")