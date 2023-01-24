#%%
import torch
import pandas as pd
from PIL import Image
from dataset import ImagesDataset
from torchvision import transforms
from classifier_1000 import TransferLearning

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
        assert torch.is_tensor(img) # immediately trigger an error if condition is false

        prediction = self.model.forward(img) 
        probability = torch.nn.functional.softmax(prediction, dim=1)
        confidence, classes = torch.max(probability, 1)
        return round(confidence.item(), 2), self.decoder[classes.item()]

if __name__ == "__main__":
    # image_size = 128
    # dataset = ImagesDataset(transform=None)
    # img, label = dataset[700]
    # img.show()
    # print(dataset.idx_to_category_name[label])

    img = ('cleaned_images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg')
    decoder = pd.read_pickle(r'decoder.pkl')    
    processor=ImageProcessor(decoder=decoder)
    prediction = processor.__prediction__(img)
    print(prediction)

#%%