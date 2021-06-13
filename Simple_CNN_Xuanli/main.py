import os
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from datasets import Dataset
from samplecnn import ZHNET
from torch.utils import data
from utils import *
from center_loss import CenterLoss
from datetime import datetime
parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.05, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default= 1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default= 0.1, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='cnn')
# misc
parser.add_argument('--eval-freq', type=int, default=5)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()


def main():
    device = torch.device('cuda:0')
    print("Creating dataset")
    train_dataset = Dataset('D:/U/Dataset/trainning')
    test_dataset = Dataset('D:/U/Dataset/test')
    trainloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    testloader = data.DataLoader(test_dataset, batch_size= 10, shuffle=False, drop_last=False)

    print("Creating model: {}".format(args.model))
    model = ZHNET(train_dataset.num_classes).to(device)
    # model_pth = './pretrained_models/Zhnet_acc-35.00_epoch-34.pth'
    # model_pth = torch.load(model_pth)
    # model.load_state_dict(model_pth['model_state_dict'])
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=train_dataset.num_classes, feat_dim=2, use_gpu=True)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_cent,
              optimizer_model,optimizer_centloss,
              trainloader, train_dataset.num_classes, epoch, device)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            acc, err = test(model, testloader,  train_dataset.num_classes, epoch, device)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            torch.save({'model_state_dict': model.state_dict()}, './pretrained_models/Zhnet_acc-{:.2f}_{:.1f}_epoch-{}.pth'.format(acc,  time.time(), epoch))
    
    
    #save pth
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, criterion_xent, criterion_cent,
          optimizer_model,optimizer_centloss,
          trainloader, num_classes, epoch, device):

    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    model.train()
    losses = AverageMeter()
    for batch_idx, data in enumerate(trainloader):
        tensor = data['tensor'].to(device)
        label = data['label'].to(device)
        features, outputs = model(tensor)
        loss_xent = criterion_xent(outputs, label)
        loss_cent = criterion_cent(features, label)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        
        for param in criterion_cent.parameters():

            param.grad.data *= (1. / args.weight_cent)

        optimizer_centloss.step()

        losses.update(loss.item(), label.size(0))
        xent_losses.update(loss_xent.item(), label.size(0))
        cent_losses.update(loss_cent.item(), label.size(0))
        
        if (batch_idx + 1) % args.print_freq == 0:
            print(" Batch {}/{}\t Loss {:.6f}".format(batch_idx+1, len(trainloader), losses.val)) 


def test(model, testloader, num_classes, epoch, device):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for datas in testloader:
            data = datas['tensor'].to(device)
            labels = datas['label'].to(device)
            # data, labels = data.to(device), labels.to(device)
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':
    main()
    




