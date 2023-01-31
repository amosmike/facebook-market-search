#%%
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from classifier import TransferLearning
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImagesDataset
import numpy
from tqdm import tqdm
import torch 
import json

def feature_extractor(model, img):
    model.eval()
    train_nodes, eval_nodes = get_graph_node_names(model)
    assert([t == e for t, e in zip(train_nodes, eval_nodes)])
    return_nodes = 'layers.fc.1'

    feat_ext = create_feature_extractor(model, return_nodes=[return_nodes])

    with torch.no_grad():
        out = feat_ext(img.unsqueeze(0)) # disable when batched
        # out = feat_ext(img)

    feat_maps = out[return_nodes].numpy().squeeze(0)

    # print(feat_maps)
    return feat_maps

if __name__ == "__main__":
    model = TransferLearning()
    image_size = 384
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop((image_size), pad_if_needed=True),
        transforms.ToTensor(),
        ]) 
    dataset = ImagesDataset(transform=transform)
    batch_size = 1  
    # dataset = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    image_id_keys = []
    image_feature_values = []

    pbar = tqdm(dataset)
    for batch in pbar:
        features, labels, image_id = batch
        feature = feature_extractor(model, features)
        image_id_keys.append(image_id)
        image_feature_values.append(feature.tolist())

    # img, labels, image_id = dataset[700]
    # feature = feature_extractor(model, img)
    # image_id_keys.append(image_id)
    # image_feature_values.append(feature.tolist())

    # print(image_id_keys)
    # print(image_feature_values)

    dictionary = dict(zip(image_id_keys, image_feature_values)) # {"image_id": [feature]}
    
    with open('image_embeddings.json', 'w') as fp:
        json.dump(dictionary, fp)

#%%
    