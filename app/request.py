#%%
import requests

response = requests.get('http://localhost:8000/healthcheck')
print(response.json())
#%%