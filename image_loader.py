#%%  
import os
import torch
import pickle
import pandas as pd
from cProfile import label
import torch.nn.functional as F
from unicodedata import category
from sklearn.metrics import accuracy_score 
from torchvision.transforms import ToTensor
from numpy import asarray, datetime_as_string
from torch.utils.data import Dataset, DataLoader

class ProductImageCategoryDataset(Dataset):

    def __init__(self, 
                labels_level: int = 0,
                root_dir: str = 'idx_to_cat.pkl',
                transform=None,
                shuffle=None,
                batch_size=None):
        super().__init__()

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"The file {self.root_dir} not found")

        self.transform = transform

        products = pd.read_pickle(self.root_dir) #, lineterminator='\n')
        # print("PRODUCTS", products)
        # products=products.sample(frac=0.1) # DEBUG - remove 
        products['category'] = products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.labels = products['category'].to_list()
        self.images = products['Image'].to_list()
        self.description = products['product_description'].to_list()
        # self.index = products['index'].to_list()
        
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
        label = int(label)
        
        image = self.images[index]

        description = self.description[index]

        # if self.transform:
        #     image = self.transform(image)
        
        return image, label, description # had brackets? 

    def __len__(self):
        return len(self.labels)

    @staticmethod 
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()

if __name__ == '__main__':
    dataset = ProductImageCategoryDataset()

# %%
