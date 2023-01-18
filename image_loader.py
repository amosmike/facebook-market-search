#%%  
from cProfile import label
from unicodedata import category
import pandas as pd
from sklearn.metrics import accuracy_score 
from torch.utils.data import Dataset, DataLoader
import os
import torch
from numpy import asarray, datetime_as_string
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pickle

class ProductImageCategoryDataset(Dataset):
    # print("CURRENT WORKING DIC", os.getcwd())

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
        # print("NUMBER OF CLASSES:", self.num_classes)
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
    
        f = open("decoder.pkl","wb")
        pickle.dump(self.decoder,f)
        f.close()

        f = open("encoder.pkl","wb")
        pickle.dump(self.encoder,f)
        f.close()

    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        # label = torch.tensor(label).long()
        label = int(label)
        
        image = self.images[index]
        # print(image)
        # image = torch.tensor(asarray(image)).float()
        # image = image.reshape(3, 64, 64) # Channels, height, width

        if self.transform:
            image = self.transform(image)
        
        return (image, label)

    def __len__(self):
        return len(self.labels)

    @staticmethod 
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()

if __name__ == '__main__':
    dataset = ProductImageCategoryDataset()
    # with open('decoder.pkl', 'rb') as handle:
    #         decoder = pickle.load(handle)
    # print(decoder)

    # with open('encoder.pkl', 'rb') as handle:
    #         encoder = pickle.load(handle)
    # print(encoder)


# %%
