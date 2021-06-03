import torch
from samplecnn import ZHNET
import torch
import numpy as np
from scipy import spatial
from datasets import Dataset
from torch.utils import data
model = ZHNET(num_classes=21).to('cpu')
model.eval()
chek = torch.load('D:/cnn model/CNN - Xuanli/pretrained_models/Zhnet_acc-16.58_epoch-4.pth')
model.load_state_dict(chek['model_state_dict'])

que_dataset = Dataset(r"D:\\try\\validation\\query")
que_lab = np.array(que_dataset.label_list)
gal_dataset = Dataset(r"D:\\try\\validation\\gallery")
gal_lab = np.array(gal_dataset.label_list)
que_loader = data.DataLoader(que_dataset, batch_size=4, shuffle=False, drop_last=False)
gal_loader = data.DataLoader(gal_dataset, batch_size= 4, shuffle=False, drop_last=False)

que_fea_list = []
gal_fea_list = []
for batch_idx, data in enumerate(que_loader):
        tensor = data['tensor']
        label = data['label']
        with torch.no_grad():
            features, outputs = model(tensor)
        que_fea_list.append(features)

for batch_idx, data in enumerate(gal_loader):
        tensor = data['tensor']
        label = data['label']
        with torch.no_grad():
            features, outputs = model(tensor)
        gal_fea_list.append(features)      
g_fea = torch.cat(gal_fea_list, 0).cpu().numpy()
q_fea = torch.cat(que_fea_list, 0).cpu().numpy()



def com_acc(label1, label2, k):
    matches = label2[:, :k]
    cor = 0
    for q_idx, q_lbl in enumerate(label1):
        cor += np.any(q_lbl == matches[q_idx, :]).item()/label1.shape[0]
    return cor
    

    
    
print('########## RESULTS ##########')



dist = spatial.distance.cdist(q_fea, g_fea, 'euclidean')
print(dist.shape)
indices = np.argsort(dist, axis=-1)
print(indices)
gal_matches = gal_lab[indices] 
for k in [1, 3, 10]:
    acc = com_acc(que_lab, gal_matches, k)
    print('--> Top-{:d} Accuracy: {:.3f}'.format(k, acc))

pass



