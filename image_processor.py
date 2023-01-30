#%%
import torch
import pandas as pd
from PIL import Image
from dataset import ImagesDataset
from torchvision import transforms
from classifier import TransferLearning
from fastapi.responses import JSONResponse

class ImageProcessor:
    def __init__(self, decoder: dict):
        '''
        '''
        self.model= TransferLearning()
        self.transform=self.model.transform
        self.state_dict=torch.load('final_model/image_model.pt')
        self.model.load_state_dict(self.state_dict)
        # self.model.eval()
        self.decoder = decoder
        
    def __prediction__(self, img):
        '''
        '''
        img=Image.open(img)
        img=self.transform(img).unsqueeze(0)
        assert torch.is_tensor(img) # immediately trigger error if condition false

        prediction = self.model.forward(img)
        probability = torch.nn.functional.softmax(prediction, dim=1)
        confidence, classes = torch.max(probability, 1)
        return self.decoder[classes.item()], round(confidence.item(), 2)

if __name__ == "__main__":
    dataset = ImagesDataset(transform=None)
    img, cat_idx = dataset[7090]
    imageid = dataset.__getimageid__(7090)
    category = dataset.idx_to_cat[cat_idx]
    path = 'cleaned_images/'
    img.show()

    decoder = pd.read_pickle(r'decoder.pkl')    
    processor=ImageProcessor(decoder=decoder)
    prediction, confidence = processor.__prediction__(path + imageid)
    print("PREDICTED CATEGORY:", prediction)
    print("CONFIDENCE:", confidence)
    print("CORRCT CATEGORY:", category)


#%%