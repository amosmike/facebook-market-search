#%%
import faiss
import json
import tensorflow as tf
import numpy as np
import torch

# ## IndexFlatL2 

# # Opening JSON file
# # f = open('image_embeddings.json')

# f = {'New York': {'3': 17, '2': 17, '2': 18},
#         'Los Angeles': {'3': 45, '1': 78, '2': 78}}
  
# # returns JSON object as a dictionary
# # data = json.load(f)



# arg = tf.convert_to_tensor(f)

# d = arg.shape

# index = faiss.IndexFlatL2(d) 

# print(d)

f = open('image_embeddings.json')
  
# returns JSON object as a dictionary
data = json.load(f)
tensor = tf.convert_to_tensor(data)

d = tensor.shape 
print(d)