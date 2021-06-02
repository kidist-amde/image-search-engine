import requests
import json
import torchvision.models as models
import IPython
import torch
from torch import nn
from torchvision  import transforms
from PIL import Image
import os
import argparse
from config import url,submission_data_path as data_path
import tqdm # use to create progress bar
# create Resnet pretrained model
# define cosine similarity 
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def cosine_similarity(query_features,gal_path):
    gal_im = Image.open(gal_path)
    gal_im = transform(gal_im).unsqueeze(0)
    gal_features = model(gal_im).detach()
    # IPython.embed();exit(1)
    return cos(query_features,gal_features)
def topk_similar_image(query_path,gal_dir,k):  
    query_im = Image.open(query_path)
    query_im = transform(query_im).unsqueeze(0)
    query_features = model(query_im)
    images_paths = []
    for file in os.listdir(gal_dir):
        if file.endswith(".jpg"):
            images_paths.append(os.path.join(gal_dir,file))
    similarity_values = {}
    for image_path in tqdm.tqdm(images_paths):
        sim = cosine_similarity(query_features,image_path)
        _,file_name = os.path.split(image_path)
        similarity_values[file_name] = sim
    topk_val = sorted(similarity_values.items(),key=lambda t:t[1],reverse=True)
    return [i[0] for i in topk_val[:k]]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
input_size = 224
transform =transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x]) if x.size(0) == 1 else x),
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
model = models.resnet50(pretrained = True, progress = True) 
# remove the output layer /Imagnet data set class 
model.fc = nn.Sequential()
model.eval()
def submit(results, url):
    res = json.dumps(results)
    response = requests.post(url, res)
    result = json.loads(response.text)
    print(f"accuracy is {result['results']}")


# data_path = "FinalPro"
# url = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/results/"

mydata = dict()
mydata['groupname'] = "TravelNet"

res = dict()
gallery_dir = os.path.join(data_path,"gallery")
for file in os.listdir(os.path.join(data_path,"query")):
    query_path = os.path.join(data_path,"query",file)
    out = topk_similar_image(query_path,gallery_dir,10)
    res[file ] = out
    
# res['<query image name>'] = ['<gallery image rank 1>', '<gallery image rank 2>', ..., '<gallery image rank 10>']
mydata["images"] = res
# print(mydata)
submit(mydata, url)
