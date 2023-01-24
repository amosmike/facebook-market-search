#%%
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import pandas as pd
import os

class ImagesDataset(Dataset):
    '''Class which defines the dataset from the folder of images in the Facebok Marketplace project'''

    def __init__(self,transform):
        super().__init__()
        self.load_dataframe()
        self.categories=self.image_df['cat_L1'].unique()
        self.all_images=self.image_df['id_x']
        self.transform=transform

        # create dict of cat_name to IDX
        self.category_name_to_idx = {
            category: cat_idx for cat_idx, category in enumerate(self.categories)
        }
        
        # create dict of IDX to cat_name
        self.idx_to_category_name= {
            value: key for
            key, value in self.category_name_to_idx.items()
        }

    def __getitem__(self, idx):
        "Selects model and features from one entry in the dataset, specified by idx"
        cwd = os.getcwd()
        image_ID=self.image_df.iloc[idx]['id_x']
        image_fp = ( cwd + '/cleaned_images/' + image_ID + '.jpg')
        # print(img_fp)
        img = Image.open(image_fp)
        if self.transform:
            img = self.transform(img)
        category_idx = self.category_name_to_idx[self.image_df.iloc[idx]['cat_L1']]
        return img, category_idx

    def __repr__(self):
        return "Nothing here yet"  
    
    def __len__(self):
        return len(self.all_images)
        
    def load_dataframe(self):
        '''loads the products csv from the Facebook Marketplace porject into a pandas Dataframe.
        Additinoally encodes the first layer of the product category as a unique int '''
        product_df = pd.read_csv('cleaned_products.csv',lineterminator='\n', index_col=0) ######
        product_df['price'] = product_df['price'].replace('[\£,]', '', regex=True).astype(float)
        image_df=pd.read_csv('images.csv',lineterminator='\n', index_col=0) #######
        self.image_df=image_df.merge(product_df, left_on ='product_id', right_on='id')
        self.image_df['cat_L1'] = [catter.split("/")[0] for catter in self.image_df['category']]
        self.image_df=self.image_df.sample(frac=0.1) # THIS IS FOR DEBUG - makes dataset much smaller!

    # def get_X_y_from_img_idx(self, idx):
    #     "Selects model and features from one entry in the dataset, specified by idx"
    #     cwd = os.getcwd()
    #     image_ID=self.image_df.iloc[idx]['id_x']
    #     image_fp = ( cwd + '/cleaned_images/' + image_ID + '.jpg')
    #     # print(img_fp)
    #     img = Image.open(image_fp)
    #     if self.transform:
    #         img = self.transform(img)
    #     category_idx = self.category_name_to_idx[self.image_df.iloc[idx]['cat_L1']]
    #     return img, category_idx

    def __show_example_image__(self,idx):
        "Displays an example image from the dataset"
        img, cat_idx=self.get_X_y_from_img_idx(idx)
        img.show()
        print('category is: ' +str(cat_idx))
    
    def __get_value_frequencies__(self):
        "Method to find relative frequencies of each product category, for dataset balancing"
        value_counts=self.image_df['cat_L1'].value_counts()
        total_n=self.image_df.shape[0]
        return (value_counts/total_n)*100

if __name__ == "__main__":
    dataset=ImagesDataset(transform=None)

    cat_to_idx=dataset.category_name_to_idx
    with open('encoder.pkl', 'wb') as handle:
        pickle.dump(cat_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    idx_to_cat=dataset.idx_to_category_name
    with open('decoder.pkl', 'wb') as handle:
        pickle.dump(idx_to_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)




# %%