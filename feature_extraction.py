import torchvision.models as models
import torch
from torch import nn
from torchvision  import transforms
from PIL import Image
import os
import argparse
import tqdm # use to create progress bar
# create Resnet pretrained model
resnet50 = models.resnet50(pretrained = True)
# define cosine similarity 
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q","--query_path",required= True,type=str,help="path to query image")
    parser.add_argument("-g","--gallery_path",required= True,type=str,help="path to gallery directory")
    args = parser.parse_args()
    def cosine_similarity(query_features,gal_path):
        gal_im = Image.open(gal_path)
        gal_im = transform(gal_im).unsqueeze(0)
        gal_features = model(gal_im)
        return cos(query_features,gal_features)
    def topk_similar_image(query_path,gal_dir,k):  
        query_im = Image.open(query_path)
        query_im = transform(query_im).unsqueeze(0)
        query_features = model(query_im)
        images_paths = []
        for folder in os.listdir(gal_dir):
            i = 0
            for file in os.listdir(os.path.join(gal_dir,folder)):
                if file.endswith(".jpg"):
                    images_paths.append(os.path.join(gal_dir,folder,file))
                    i +=1
                    if i >= 2:
                        break
        similarity_values = {}
        for image_path in tqdm.tqdm(images_paths):
            sim = cosine_similarity(query_features,image_path)
            similarity_values[image_path] = sim
        topk_val = sorted(similarity_values.items(),key=lambda t:t[1],reverse=True)
        return [i[0] for i in topk_val[:k]]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),normalize])
    model = models.resnet50(pretrained = True, progress = True) 
    # remove the output layer /Imagnet data set class 
    model.fc = nn.Sequential()
    gal_dir =  args.gallery_path
    query_path = args.query_path
    out = topk_similar_image(query_path,gal_dir,3)
    print(out)

if __name__ == "__main__":
    main()
    