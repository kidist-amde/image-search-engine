import torch
from samplecnn import ZHNET
import torch
import numpy as np
from scipy import spatial
from datasets import Dataset
from torch.utils import data
model = ZHNET(num_classes=21).to('cpu')
model.eval()
chek = torch.load('D:/cnn model/CNN_Xuanli/pretrained_models/Zhnet_acc-66.7_1623581954.6_epoch-49.pth')
model.load_state_dict(chek['model_state_dict'])


que_dataset = Dataset("D:/try/test/query")
que_lab = np.array(que_dataset.label_list)
que_path = que_dataset.img_list
gal_dataset = Dataset("D:/try/test/gallery")
gal_lab = np.array(gal_dataset.label_list)
gal_path = gal_dataset.img_list
que_loader = data.DataLoader(que_dataset, batch_size=4, shuffle=False, drop_last=False)
gal_loader = data.DataLoader(gal_dataset, batch_size= 4, shuffle=False, drop_last=False)

que_fea_list = []
que_pre = np.asarray([])
gal_fea_list = []
gal_pre = np.asarray([])

for batch_idx, data in enumerate(que_loader):
        tensor = data['tensor']
        label = data['label']
        with torch.no_grad():
            features, outputs = model(tensor)
            predictions = outputs.data.max(1)[1].numpy()
            print(type(predictions))

        que_fea_list.append(features)
        que_pre = np.concatenate((que_pre, predictions))

print(que_pre)
        # print(np.concatenate(np.vstack(que_pre)))
        # print(que_pre.shape)
        # que_pre = que_pre.reshape((15, 1))

for batch_idx, data in enumerate(gal_loader):
        tensor = data['tensor']
        label = data['label']
        with torch.no_grad():
            features, outputs = model(tensor)
            predictions = outputs.data.max(1)[1].numpy()
            
        gal_fea_list.append(features)
        gal_pre = np.concatenate((gal_pre, predictions))
print(gal_pre)

# pred_y = torch.max(predict, 1)[1].numpy()
# label_y = torch.max(label, 1)[1].data.numpy()
# accuracy = (pred_y == label_y).sum() / len(label_y)

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
# print(dist.shape)
indices = np.argsort(dist, axis=-1)
print(indices)

gal_matches = gal_pre[indices]

# for i in range(len(que_pre)):
#     ind = indices[i:,]
#     gal = gal_pre[ind]
#     gal_matches.append(gal)

# for k in [1, 3, 10]:
#     acc = com_acc(que_pre, gal_matches, k)
#     print('--> Top-{:d} Accuracy: {:.3f}'.format(k, acc))

# pass

res = {}
gal_match = np.array(gal_path)[indices]
for i in range(len(que_path)):
    match = []
    for j in gal_match[i,:10]:
        match.append(j.replace("D:/try/test/gallery\\1\\", ""))
    res[que_path[i].replace('D:/try/test/query\\1\\', "")] = match

print(res)

