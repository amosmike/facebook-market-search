import requests

response = requests.get('http://127.0.0.1:8000/healthcheck')
print(response.json())

# response = requests.post('http://127.0.0.1:8000/predict/image/')
# print(response.json())