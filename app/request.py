#%%
import requests
from PIL import Image
import base64
import json
# response = requests.get('http://127.0.0.1:8080/healthcheck')
# img = Image.open('0a1baaa8-4556-4e07-a486-599c05cce76c.jpg').tobytes()
# img.show()
# print(type(img))

api = 'http://127.0.0.1:8080/predict/image'

image_file = '0a1baaa8-4556-4e07-a486-599c05cce76c.jpg'

with open(image_file, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
payload = json.dumps({"image": im_b64, "other_key": "value"})

response = requests.post(api, data=payload, headers=headers)
# try:
data = response.json()     
print(data)                
# except requests.exceptions.RequestException:
#     print(response.text)

#%%