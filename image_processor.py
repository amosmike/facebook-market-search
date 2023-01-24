#%%
import torch
import pickle
from PIL import Image
from torchvision import transforms
from classifier_1000 import TransferLearning

class ImageProcessor:

    def __init__(self, idx_to_cat: dict):
        '''class constructor - inputs are dicts to transform indices to categories'''
        self.model= TransferLearning()
        self.transform=self.model.transform
        self.state_dict=torch.load('final_models/image_model.pt ')
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        
        self.idx_to_cat = idx_to_cat
        
    def __prediction__(self,image_fp):
        '''function to generate a prediction based on an input image'''
        img=Image.open(image_fp) 
        img=self.transform(img).unsqueeze(0)
        assert torch.is_tensor(img)
        prediction = self.model.forward(img) 
        probs = torch.nn.functional.softmax(prediction, dim=1)
        conf, classes = torch.max(probs, 1)
        return conf.item(), self.idx_to_cat[classes.item()]

# image_fp = ('cleaned_images/0c2f81f8-7d98-42e2-9d7d-836335fa08df.jpg')

# with open('idx_to_cat.pickle', 'rb') as handle:
#     idx_to_cat=pickle.load(handle)

# processor=ImageProcessor(idx_to_cat = idx_to_cat)
# processor.get_prediction(image_fp)

# %%