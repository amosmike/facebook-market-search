#%%
import requests
from PIL import Image
import base64
import json
from api import ImageClassifier
from classifier import TransferLearning


response = requests.get('http://localhost:8080/healthcheck')
print(response.json())
# img = Image.open('0a1baaa8-4556-4e07-a486-599c05cce76c.jpg').tobytes()
# img.show()
# print(type(img))

# api = 'http://127.0.0.1:8080/predict/image'
# url = 'http://127.0.0.1:8080/files/'

# img = '0a1baaa8-4556-4e07-a486-599c05cce76c.jpg'

# model = ImageClassifier(TransferLearning)
# transform=model.transform
# print(type(img))
# image_file = transform(img).unsqueeze(0)

# with open(image_file, "rb") as f:
#     # im_bytes = f.read()        
#     png_encoded = base64.b64decode(f.read())

# im_b64 = base64.b64encode(im_bytes).decode("utf8")

# with requests.Session() as s:
#     r=s.post(url,files={'file': png_encoded})
#     print(r.json())

# headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
# payload = json.dumps({"image": im_b64, "other_key": "value"})

# response = requests.post(api, files=image_file)
# # try:
# data = response.json()     
# print(data)                
# except requests.exceptions.RequestException:
#     print(response.text)



#%%