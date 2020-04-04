import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from sklearn.model_selection import train_test_split
import os
import cv2
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
import time
import vgg16M 

Datadir = "F:\Datasetcas"
Kategori = ["ghibli","kyoani2","other"]
img_size = 224
dataall = []

for category in Kategori:
    path = os.path.join(Datadir, category)
    class_all = Kategori.index(category)
        
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (img_size, img_size), interpolation = cv2.INTER_CUBIC)
        dataall.append([new_array, class_all])

random.shuffle(dataall)        

X = []
y = []

for fitur, label in dataall:
    X.append(fitur)
    y.append(label)
    
X = np.array(X).astype("uint8")
y = np.array(y).astype("uint8")

def shuffle(features, labels, proportions):
    ratio = int(features.shape[0]/proportions)
    X_train = features[ratio:, :]
    X_test = features[:ratio, :]
    y_train = labels[ratio:]
    y_test = labels[:ratio]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = shuffle(X, y, 3)

model = vgg16()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)

for epoch in range(3):
    adjust_learning_rate(optimizer, epoch)
    train(X_train, y_train, model, criterion, optimizer, epoch)
    prec1 = validate(X_test, y_test, model, criterion)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_prec1,
        }, is_best, filename = 'checkpoint.pth.tar')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    lr = 0.05 * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(X_train, y_train, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    end = time.time()
    
    for i in enumerate(X_train):
        data_time.update(time.time() - end)
        input = X_train[i].cuda()
        target = y_train[i].cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(X_train), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

def validate(X_test, y_test, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()
    for i in enumerate(X_test):
        input = X_test[i].cuda()
        target = y_test[i].cuda()
        
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(X_train), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg   

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)



