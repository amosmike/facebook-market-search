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
        self.model=TransferLearning()
        self.transform=self.model.transform
        self.state_dict=torch.load('final_model/image_model.pt')
        self.model.load_state_dict(self.state_dict)
        # self.model.eval()
        self.decoder = decoder
        
    def prediction(self, img):
        img=Image.open(img)
        img=self.transform(img).unsqueeze(0)
        assert torch.is_tensor(img) # immediately trigger error if condition false

        prediction = self.model.forward(img)
        prediction_list = prediction.tolist()[0]
        prediction_list = ['%.2f' % elem for elem in prediction_list]

        probability = torch.nn.functional.softmax(prediction, dim=1)
        probability_list = probability.tolist()[0]
        probability_list = ['%.2f' % elem for elem in probability_list]

        confidence, category = torch.max(probability, 1)
        print(round(confidence.item(), 2))

        pos_cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        pos_cat_str = []
        for i in pos_cat_idx:
            pos_cat_str.append(decoder[i])

        confidence_dic = dict(zip(pos_cat_str, probability_list))
        
        return self.decoder[category.item()], round(confidence.item(), 2), confidence_dic

if __name__ == "__main__":
    dataset = ImagesDataset(transform=None)
    img, category_idx, image_id = dataset[7090]
    imageid = dataset.getimageid(7090)
    category = dataset.idx_to_cat[category_idx]
    path = 'cleaned_images/'
    # img.show()

    decoder = pd.read_pickle(r'decoder.pkl')  
    processor=ImageProcessor(decoder=decoder)
    prediction, confidence, confidence_dic = processor.prediction(path + imageid)
    # print("PREDICTED CATEGORY:", prediction)
    # print("CONFIDENCE:", confidence)
    # print("CORRCT CATEGORY:", category)


#%%