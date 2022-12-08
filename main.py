#two model not shared


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

import cv2


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

def imshow(img,num):
    # 非正規化する
    #img = img / 2 + 0.5
    # torch.Tensor型からnumpy.ndarray型に変換する
    #print(type(img)) # <class 'torch.Tensor'>
    #print(img.shape)
    npimg = img.to('cpu').detach().numpy().copy()
    
    #print(type(npimg))    
    # 形状を（RGB、縦、横）から（縦、横、RGB）に変換する
    #print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    #print(npimg.shape)
    #npimg.save('./result/cifar10_vis'+dt_now_str+'.jpg',npimg*255)
    
    #print(npimg)
    npimg=cv2.cvtColor(npimg,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_'+dt_now_str+str(eps)+'_EPS.jpg',npimg*255)
    
 

#Adversarial example parameters#

model_name="resnet50"
atk_name="PGD"
#model_file="ckpt2"
model_file="ckpt_STL_128"
alpha=2/255
step=4
cam_mode = "eigen"  #grad_cam / eigen
eps_list = [0/255,2/255,4/255,8/255,16/255,32/255,64/255]
train_con = 0




for eps in eps_list:
    print("****************************EPS:",eps,"*********************************")
    if train_con == 0:

        train_data_set = MyDataset_train(model_file,model_name,cam_mode)
        train_d = train_data_set()
        #train_x,train_y = train_d
        #train_data = torch.utils.data.TensorDataset(train_x, train_y)
        train_dataloader = DataLoader(train_d, batch_size=128, shuffle=True)
        train_con = 1


    val_data_set = MyDataset_val(model_name,atk_name,model_file,eps,alpha,step,cam_mode)
    val_d = val_data_set()

    #print(len(val_d))
    #val_x,val_y = val_d
    #val_data = torch.utils.data.TensorDataset(val_x, val_y)

    n_samples = len(val_d)
    val_num = int(len(val_d)*0.8)
    test_num = n_samples - val_num
    validation_data,test_data = torch.utils.data.random_split(val_d,[val_num,test_num])
    validation_dataloader = DataLoader(validation_data, batch_size=100, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False)



    names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    model = models.resnet50(pretrained = True)
    model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0005)
    #optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.00005)
    #CIFAR10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()

    print("****************************************************")
    print(device)
    num_epochs = 100
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    
    #######################################################
    dataiter = iter(validation_dataloader)
    it_img,it_labels = dataiter.next()
    imshow(torchvision.utils.make_grid(it_img),eps)
    #####################################################"##"
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        
        model.train()
        for imgs, labels in tqdm(train_dataloader):
            
            imgs = imgs.to(device)

            labels = labels.to(device)
    
            labels = torch.squeeze_copy(labels)
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
            val_labels = torch.squeeze_copy(val_labels)
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
        
    model.eval()
    test_running_loss = 0.0
    test_running_acc = 0.0
    for test_imgs, test_labels in tqdm(test_dataloader):

        test_imgs = test_imgs.to(device)
        test_labels = test_labels.to(device)
        test_labels = torch.squeeze_copy(test_labels)
        test_output = model(test_imgs)
        test_loss = criterion(test_output, test_labels)
        test_running_loss += test_loss.item()
        test_pred = torch.argmax(test_output, dim=1)
        test_running_acc += torch.mean(test_pred.eq(test_labels).float())
        
    test_running_loss /= len(test_dataloader)
    test_running_acc /= len(test_dataloader)
    print("-------------------test model------------------------")
    print("eps{} :loss: {}, acc: {}, ".format(eps,test_running_loss, test_running_acc))
