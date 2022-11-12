from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import models,datasets, transforms
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime
import pytz
import os
import pandas as pd
import torch.optim as optim

from transform import grad_transforms_train
from transform import grad_transforms_val
from dataset import MyDataset_train
from dataset import MyDataset_val

#tensorboard pop install tensorboard 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./logs")

#device CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#PATH
os.makedirs('model',exist_ok=True)
model_PATH = './model'

#datetime
dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

dt_now_str = str(dt_now)
dt_now=dt_now.strftime('/%m_%d_%H:%M')





'''
#data loaD  TRANSFORMS.PY DATASET.PY
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform("11_07_16:01epochs200","resnet"))
val_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
validation_dataloader = DataLoader(val_data, batch_size=128, shuffle=False)
names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
'''
#Adversarial example parameters#

model_name="resnet50"
atk_name="FGSM"
model_file="11_07_19:56epochs200"
eps=2/255
alpha=2/255
step=4

#data load  TRANSFORMS.PY DATASET.PY

val_data_set = MyDataset_val(model_name,atk_name,model_file,eps,alpha,step)
val_data = val_data_set()

val_num = int(len(val_data))*0.8
test_num = int(len(val_data))*0.2
validation_data,test_data = torch.utils.data.random_spliit(val_data,[val_num,test_num])

train_data_set = MyDataset_train(model_file,model_name)
train_data = train_data_set()


print("*********************",len(train_data))
val_num = int(len(val_data))*0.8
test_num = int(len(val_data))*0.2
validation_data,test_data = torch.utils.data.random_split(val_data,[val_num,test_num])

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

model = models.resnet50(pretrained = True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

print("****************************************************")
print(device)
num_epochs = 100
losses = []
accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    
    model.train()
    for imgs, labels in tqdm(train_dataloader):
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        output = model(imgs)
        loss = criterion(output, labels)
        
        loss.backward()
        running_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        optimizer.step()
    
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    
    losses.append(running_loss)
    accs.append(running_acc)
    #
    # validation loop
    #
    model.eval()

    val_running_loss = 0.0
    val_running_acc = 0.0
    for val_imgs, val_labels in tqdm(validation_dataloader):
        
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        
        val_output = model(val_imgs)
        val_loss = criterion(val_output, val_labels)
        val_running_loss += val_loss.item()
        val_pred = torch.argmax(val_output, dim=1)
        val_running_acc += torch.mean(val_pred.eq(val_labels).float())
    
    val_running_loss /= len(validation_dataloader)
    val_running_acc /= len(validation_dataloader)
   
    val_losses.append(val_running_loss)
    val_accs.append(val_running_acc)
    
    writer.add_scalar("running_loss", running_loss, epoch)
    writer.add_scalar("val_running_loss", val_running_loss, epoch)
    writer.add_scalar("val_running_acc", val_running_acc, epoch)
    writer.add_scalar("running_acc", running_acc, epoch)



    print("epoch: {}, loss: {}, acc: {}, \
     val loss: {}, val acc: {}".format(epoch, running_loss, running_acc, val_running_loss, val_running_acc))


