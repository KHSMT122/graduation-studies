import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms,models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline
import datetime
import pytz

import os

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

#data aug
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

#data load
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")



#Resnet model
model = models.resnet50(pretrained = True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
model = model.to(device)


#optimizer & loss
optimizer=optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()


#running
print("****************************************************")
print(device)
num_epochs = 1
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
    
    print("epoch: {}, loss: {}, acc: {}, \
     val loss: {}, val acc: {}".format(epoch, running_loss, running_acc, val_running_loss, val_running_acc))
print("save model")
torch.save(model.state_dict(),model_PATH+dt_now+"epochs"+str(num_epochs)+".pth")


