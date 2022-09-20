from cProfile import label
from unicodedata import category
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import os
import torch
from numpy import asarray, datetime_as_string
from torchvision.transforms import ToTensor
import torch.nn.functional as F

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

        image = torch.tensor(asarray(image)).float()
        image = image.reshape(3, 64, 64) # Channels, height, width

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

if __name__ == '__main__':
    dataset = ProductImageCategoryDataset()