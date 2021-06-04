from __future__ import print_function
from __future__ import division
import torch,torchvision, pandas as pd , numpy as np, os,time,copy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import IPython
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import warnings 
warnings.filterwarnings("ignore") 
from config import train_path,val_path, best_model_path ,best_state_path 
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class  SearchEngineDataset(Dataset):
    def __init__(self,folder,label_mapping = None,transform = None):
        self.folder = folder
        self.label_mapping = label_mapping
        self.loaded_data ={}
        self.transform = transform
        self.load_data()# Create training and validation datasets
    def load_data(self):
        self.labels = []
        self.file_paths = []
        for label in os.listdir(self.folder):
            for image_file in os.listdir(os.path.join(self.folder,label)):
                if image_file.lower().endswith(".jpg") or image_file.lower().endswith(".jpeg") or \
                    image_file.lower().endswith(".png"):
                    self.labels.append(label)
                    self.file_paths.append(os.path.join(self.folder,label,image_file))
        # get unique label mapping
        if self.label_mapping is None:
            classes = set(self.labels)  
            classes = list(classes)
            classes.sort()
            self.label_mapping = {classes[i]:i for i in range(len(classes))}
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # if idx in self.loaded_data:
        #     return self.loaded_data[idx]
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        image  = Image.open(file_path)
        if not image.mode=="RGB":
          image = image.convert("RGB")
          filename, _ = os.path.splitext(file_path)
          new_file_path = filename+".jpg"
          image.save(new_file_path)
          self.file_paths[idx] = new_file_path

        label = self.label_mapping[label]
        if self.transform is not None:
            image = self.transform(image)
        # self.loaded_data[idx] = image,label
        return image,label
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def train_model(model,dataloader,optimizer,criterion):
    model.layer4.train()
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for inputs, labels in tqdm(dataloader,total = len(dataloader)):
        # move the images to the  choosen  dvice (GPU or cpu)
        inputs = inputs.to(device) 
        labels = labels.to(device)
        # compute the o/p
        outputs = model(inputs)
        # compute the loss
        loss = criterion(outputs, labels)
        # make prediction
        _, preds = torch.max(outputs, 1)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradients 
        loss.backward()
        # update the weight
        optimizer.step()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss,epoch_acc
        
def evaluate_model(model,dataloader,criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        # Iterate over data.
        for inputs, labels in tqdm(dataloader,total = len(dataloader)):
            # move the images to the  choosen  dvice (GPU or cpu)
            inputs = inputs.to(device) 
            labels = labels.to(device)
            # compute the o/p
            outputs = model(inputs)
            # compute the loss
            loss = criterion(outputs, labels)
            # make prediction
            _, preds = torch.max(outputs, 1)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss,epoch_acc
def training_loop (model,dataloaders,optimizer,criterion,num_epochs):
    #to start from where the excusion stoped
    if os.path.exists(best_state_path):
      state = (torch.load(best_state_path))
      model.load_state_dict(state["model_state"])
      best_acc = state["best_acc"]
      optimizer = state["optimizer"]
      start_epoch = state["epoch"]
    else:
        best_acc = 0
        start_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    val_accs = []
    for epoch in range(start_epoch,num_epochs):
        train_loss,train_acc = train_model(model,dataloaders["train"],optimizer,criterion)
        val_loss,val_acc = evaluate_model(model,dataloaders["val"],criterion)
        val_accs.append(val_acc)
        if epoch%10==0:
            print("Epoch: {} train loss: {:.4f} train acc: {:.4f} val loss: {:.4f} val acc: {:.4f}"\
                .format(epoch,train_loss,train_acc,val_loss,val_acc))
        # save best state
        if best_acc < val_acc:
            print("found best model==> improved from {:.4f} to {:.4f}".format(best_acc,val_acc))
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state":best_model_state,"epoch":epoch,"optimizer":optimizer,\
                "best_acc":best_acc},best_state_path)

    model.load_state_dict(best_model_state)
    return model,val_accs
    
def main():
    # train_path = "all_data/training"
    # val_path = "all_data/validation/query"
    input_size = 224
    num_epochs = 200
    batch_size = 32
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                                        saturation=0.8, hue=0.2)],p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}
    train_dataset = SearchEngineDataset(train_path,transform = data_transforms["train"])
    val_dataset = SearchEngineDataset(val_path,label_mapping=train_dataset.label_mapping,
                                      transform = data_transforms["val"])
    image,label = train_dataset[0]
    image = image - image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] 
    image = image /image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] 
    # plt.imshow(image.permute(1,2,0))
    # plt.show()
    # IPython.embed()
    # print(len(train_dataset))
    # Create training and validation datasets
    image_datasets = {"train":train_dataset,"val":val_dataset}
    
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
              batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
        
    model = models.resnet18(pretrained="imagenet")
    set_parameter_requires_grad(model, feature_extracting = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.label_mapping))
    model = model.to(device)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # print(model)
    # parameters are being optimized
    optimizer = optim.SGD(model.layer4.parameters(), lr=0.001, momentum=0.9)
    # Setup the loss fun
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    model, hist = training_loop(model, dataloaders_dict, optimizer, criterion,num_epochs=num_epochs)
    torch.save(model,best_model_path)
    


if __name__ == "__main__":
    main()