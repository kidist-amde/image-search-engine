
import os,numpy as np ,pandas as pd ,IPython,shutil
from sklearn.model_selection import train_test_split
from config import download_data_path as data_path,split_data_path as split_data

def main():
    # data_path = "downloads"
    # split_data = "train_val"
    if not os.path.exists(split_data):
        os.mkdir(split_data)
    labels = []
    file_paths = []
    for label in os.listdir(data_path):
        for image_file in os.listdir(os.path.join(data_path,label)):
            if image_file.lower().endswith(".jpg") or image_file.lower().endswith(".jpeg") or \
                image_file.lower().endswith(".png"):
                labels.append(label)
                file_paths.append(os.path.join(data_path,label,image_file))
    train_paths,val_paths, train_labels, val_labels = train_test_split(file_paths, labels,\
        stratify= labels, test_size=0.2, random_state=42)
    for t in ["train","val"]:
        if not os.path.exists(os.path.join(split_data,t)):
            os.mkdir(os.path.join(split_data,t))
    for i in  range(len(train_paths)):
        # IPython.embed()
        label_folder = os.path.join(split_data,"train",train_labels[i])
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)
        root_folder,file_name = os.path.split(train_paths[i])
        src_path  = train_paths[i]
        dst_path = os.path.join(label_folder,file_name)
        shutil.copy(src_path,dst_path)
        
    for i in  range(len(val_paths)):
        # IPython.embed()
        label_folder = os.path.join(split_data,"val",val_labels[i])
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)
        root_folder,file_name = os.path.split(val_paths[i])
        src_path  = val_paths[i]
        dst_path = os.path.join(label_folder,file_name)
        shutil.copy(src_path,dst_path)
     
        
    
if __name__ == "__main__":
    main()