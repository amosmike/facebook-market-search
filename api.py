import os
import pickle
import json

import uvicorn
import boto3
import botocore
import numpy as np

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch.nn as nn
import torch
import pandas as pd
from classifier import TransferLearning
from fastapi import UploadFile
from fastapi import File

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

@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    img = Image.open(image.file)
    img = transform(img).unsqueeze(0)
    assert torch.is_tensor(img)

    prediction = model.forward(img)
    probability = torch.nn.functional.softmax(prediction, dim=1)
    confidence, classes = torch.max(probability, 1)
    prediction = decoder[classes.item()]
    confidence = round(confidence.item(), 2)

    return JSONResponse(content={
    "Category": prediction, # Return the category here
    "Probability": confidence # Return a list or dict of probabilities here
    # "Classes": classes # Return a list or dict of probabilities here
        })

if __name__ == '__main__':
  uvicorn.run(app, host='127.0.0.1', port='8000') # uvicorn api:app --reload
