from cProfile import label
from unicodedata import category
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import os
import torch
from numpy import asarray, datetime_as_string
from torchvision.transforms import ToTensor
import torch.nn.functional as F


# dataset = ImageDataset(labels_level=0, root_dir='saved_data/data_all.pkl', transform=my_transform)

class ProductImageCategoryDataset(Dataset):
    def __init__(self, 
                labels_level: int = 0,
                root_dir: str = 'saved_data/data_all.pkl',
                transform=None,
                shuffle=None,
                batch_size=None):
        super().__init__()

        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"The file {self.root_dir} not found")

        self.transform = transform

        products = pd.read_pickle(self.root_dir) #, lineterminator='\n')
        products['category'] = products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.labels = products['category'].to_list()
        self.images = products['Image'].to_list()
        
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        label = torch.tensor(label).long()
        
        image = self.images[index]

        # labels = torch.tensor(self.y.iloc[index]).long()
        # features = torch.tensor(features).float()

        image = torch.tensor(asarray(image)).float()
        image = image.reshape(3, 64, 64) # Channels, height, width


        # print(image.dtype)

        label = int(label)

        if self.transform:
            image = self.transform(image)
        
        return (image, label)

    def __len__(self):
        return len(self.labels)

    @staticmethod 
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()

dataset = ProductImageCategoryDataset()

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

def train(model, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    # writer = SummaryWriter()

    # batch_idx = 0

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            # prediction = model(features) # WRONG
            # example = dataset[batch]
            # print(label.shape)
            # print(features.shape)

            loss = F.cross_entropy(prediction, labels) #binary_cross_entropy
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()

class NN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Neural Network
        self.convolution = torch.nn.Conv2d(3, 6, 9) # This is a CNN, no? 
        self.pooling = torch.nn.AvgPool2d(3) # 3 = no. keras = 'search block' used for reduction
        self.flattern = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear(1944, 13) #(No. features, no. layers) # ERROR


        # Improve model by increasing conv layers 
        # relu for activation 

    def forward(self, X):
        # print(X.shape)
        X = self.convolution(X)
        X = self.pooling(X)
        X = self.flattern(X)
        X = self.linear_layer(X)

        # X = F.relu(X)
        # X = self.linear_layer2(X)
        return X

if __name__ == '__main__':
    dataset = ProductImageCategoryDataset()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=8)
    model = NN()
    train(model)

# class Module():
#     def __init__(self) -> None:
#         pass

#     def __call__():

#         self.forward()

#     def forward():
#         raise NotImplementedError

########## Uncomment below ##########

# class CNN(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.layers = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 8, 9), # input, output, kernal size. Why output 8? 
#             torch.nn.Flatten(), # Flatern
#             torch.nn.Linear(87678, 13), # Simplify further with linear layers
#             torch.nn.Softmax() # Turn into probabilities
#             )

#     def forward(self, features):
#         return self.layers(features)
 
# model = CNN()

# for batch in train_loader:
#     features, labels = batch
#     prediction = model(features)
#     break


# prediction = model(features)