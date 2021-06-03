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
parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=40)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
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
    print("Creating dataset")
    train_dataset = Dataset('D:/try/training')
    test_dataset = Dataset('D:/try/training')
    trainloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    testloader = data.DataLoader(test_dataset, batch_size= 10, shuffle=False, drop_last=False)

    print("Creating model: {}".format(args.model))
    model = ZHNET(train_dataset.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion,
              optimizer_model,
              trainloader, train_dataset.num_classes, epoch)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            acc, err = test(model, testloader,  train_dataset.num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            

            torch.save({'model_state_dict': model.state_dict()}, './pretrained_models/Zhnet_acc-{:.2f}_epoch-{}.pth'.format(acc, epoch))
    #save pth
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, criterion,
          optimizer_model,
          trainloader, num_classes, epoch):
    model.train()
    losses = AverageMeter()
    for batch_idx, data in enumerate(trainloader):
        tensor = data['tensor']
        label = data['label']
        features, outputs = model(tensor)
        loss = criterion(outputs, label)  # + loss1(features)
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        losses.update(loss.item(), label.size(0))
        if (batch_idx+1) % args.print_freq == 0:
            print(" Batch {}/{}\t Loss {:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(model, testloader, num_classes, epoch):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data in testloader:
            tensor = data['tensor']
            label = data['label']
            features, outputs = model(tensor)
            predictions = outputs.data.max(1)[1]
            total += label.size(0)
            correct += (predictions == label.data).sum()
    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':
    main()
    




