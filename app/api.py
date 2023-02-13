#%%
from flask import Flask, request, jsonify, abort
from fastapi.responses import JSONResponse
from classifier import TransferLearning
from fastapi import UploadFile
from fastapi import FastAPI
from fastapi import File
from PIL import Image
import pandas as pd
import numpy as np
import uvicorn
import base64
import torch
import io

class ImageClassifier(TransferLearning):
    def __init__(self,
                 decoder: dict = None):
        super().__init__()
        self.decoder = pd.read_pickle(r'decoder.pkl')

    def forward(self, image):
        x = self.layers(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            prediction = self.forward(image)
            return prediction

    def predict_proba(self, image):
        with torch.no_grad():
            prediction = self.model.forward(image)
            probability = torch.nn.functional.softmax(prediction, dim=1)
            return probability

    def predict_classes(self, image):
        with torch.no_grad():
            prediction = self.forward(image)
            probability = torch.nn.functional.softmax(prediction, dim=1)
            confidence, classes = torch.max(probability, 1)
            return classes

try:
    model = ImageClassifier()
    transform=model.transform
    state_dict = torch.load('final_model/image_model.pt')
    model.load_state_dict(state_dict=state_dict)

except:
    raise OSError("NO IMAGE MODEL FOUND")

try:
    class ImageProcessor:
        def __init__(self): 
            self.transform=model.transform
        
        def transform_image(self,img):
            return self.transform(img).unsqueeze(0)

    decoder = pd.read_pickle(r'decoder.pkl')
    processor=ImageProcessor

except:
    raise OSError("NO IMAGE PROCESSOR FOUND")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  return {"message": msg}

@app.route("/predict/image", methods=['POST'])
# @app.post('/predict/image', methods=['POST'])
def predict_image(image: UploadFile = File(...)):
    if not request.json or 'image' not in request.json: 
        abort(400)
    
    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)

    # img = Image.open(image, mode='rb')
    # img = transform(img).unsqueeze(0)
    # assert torch.is_tensor(img)

    # prediction = model.forward(img)
    prediction = model.forward(img_arr)

    prediction_list = prediction.tolist()[0]
    prediction_list = ['%.2f' % elem for elem in prediction_list]

    probability = torch.nn.functional.softmax(prediction, dim=1)
    probability_list = probability.tolist()[0]
    probability_list = ['%.2f' % elem for elem in probability_list]

    confidence, category = torch.max(probability, 1)
    category = decoder[category.item()]
    confidence = round(confidence.item(), 2)

    categorys = list(decoder.values())

    confidence_dic = dict(zip(categorys, probability_list))

    return JSONResponse(content={
    "Category": category, # Return the category here
    "Confidence": confidence, # Returns probability of chosen category
    "Probabilitys": confidence_dic # Returns dict of probabilities
    })

if __name__ == '__main__':
  uvicorn.run(app, host='127.0.0.1', port='8080') # uvicorn api:app --reload
  # http://127.0.0.1:8080/docs
