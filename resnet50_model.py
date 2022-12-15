#two heads resnet50 backbone problem:target layers

















import cv2
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
from torchvision.models import feature_extraction
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models._utils import IntermediateLayerGetter


import torch
import torchvision
from PIL import Image
from torchvision import transforms
import os
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

import torchattacks
from torchattacks import PGD,FGSM,CW

import torch.backends.cudnn as cudnn

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM,EigenCAM


model = models.resnet50(pretrained = True)
names = torchvision.models.feature_extraction.get_graph_node_names(model)

print(model)
'''
print(names)


def print_model(module, name="model", depth=0):
    if len(list(module.named_children())) == 0:
        print(f"{' ' * depth} {name}: {module}")
    else:
        print(f"{' ' * depth} {name}: {type(module)}")

    for child_name, child_module in module.named_children():
        if isinstance(module, torch.nn.Sequential):
            child_name = f"{name}[{child_name}]"
        else:
            child_name = f"{name}.{child_name}"
        print_model(child_module, child_name, depth + 1)


print_model(model)
'''

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
    
 

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)

        self.backbone = IntermediateLayerGetter(self.backbone,return_layers={"avgpool":"out"})
        self.flatten =  nn.Flatten()
        #self.backbone = create_feature_extractor(self.backbone, ["flatten"])
        self.fc1 = nn.Sequential(
                nn.Linear(2048, 10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 10)
        )
    def forward(self, x, change_head='normal'):
        #x = self.backbone(x)["out"]
        x = self.backbone(x)["out"]
        x = self.flatten(x)
        if change_head == "normal":
            x = self.fc1(x)
        elif change_head == "g_normal":
            x = self.fc2(x)
        else:
           raise NotImplementedError
        return x
    
    #device CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"


#datetime
dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
dt_now_str = str(dt_now)
dt_now=dt_now.strftime('/%m_%d_%H:%M')

#data load

train_batch = 128
val_batch = 100

transform = transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),

        ])       

trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform,
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=train_batch,
    shuffle=True,
    num_workers=2,
    pin_memory=True
    )

val_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
    )
n_samples = len(trainset)
val_size = int(len(trainset) * 0.8)
test_size = n_samples - val_size
testset, valset = torch.utils.data.random_split(
        trainset,
        [test_size, val_size]
        )

val_load = torch.utils.data.DataLoader(
    val_set,
    batch_size=val_batch,
    shuffle=False,
    num_workers=2,
    pin_memory=True
    )

test_load = torch.utils.data.DataLoader(
    testset,
    batch_size=val_batch,
    shuffle=False,
    num_workers=2,
    pin_memory=True
    )

names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


#fx_model = models.resnet50(pretrained = True)
#fx_model.fc = nn.Linear(2048,10)
#print(fx_model)

fx_model = Network()

#x_model = fx_transform(fx_model)
fx_model = fx_model.to("cuda")
print(fx_model)

'''
optimizer = optim.SGD(
        fx_model.parameters(),
        lr=0.05,
        momentum=0.9,
        )

'''
optimizer = optim.Adam(fx_model.parameters(),lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=50)

criterion = nn.CrossEntropyLoss()


#atk = PGD(fx_model,eps=4/255,alpha = 8/255,steps=4, random_start=False)
#atk = FGSM(fx_model,eps=2/255)
#atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])




num_epochs = 100
losses = []
accs = []
val_losses = []
val_accs = []
#####################################################################################
atk_name="FGSM"
model_file="ckpt2"
eps=64/255
alpha=2/255
step=4
cam_mode = "grad_cam"
#####################################################################################
for epoch in range(num_epochs):
    fx_model.train()

    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in tqdm(trainloader):
        c_imgs = imgs
        c_labels = labels

       

        norm = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        imgs = norm(imgs)     #<torch.float32>
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = fx_model(imgs,"normal")
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        running_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        
        optimizer.step()
        #cam_img forward

        fx_model.eval()

        c_labels = c_labels.to(device)
        norm = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_img = norm(c_imgs)     #<torch.float32>
        rgb_img = c_imgs     
        fx_model.eval()
        target_layers = [
            getattr(fx_model.backbone.layer1, '2'), 
            getattr(fx_model.backbone.layer2, '3'), 
            getattr(fx_model.backbone.layer4, '2'),
            getattr(fx_model.backbone.layer3, '5')]
        '''
        target_layers = [
            fx_model.backbone.layer1[-1],
            fx_model.backbone.layer2[-1],
            fx_model.backbone.layer3[-1],
            fx_model.backbone.layer4[-1]
            ]  '''
        
        input_img = input_img.to("cuda")
        #input_img = input_img.unsqueeze(0) 
       

        if cam_mode == "grad_cam":
            with GradCAM(model=fx_model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
                cam.batch_size=train_batch
                grayscale_cam = cam(
                #input_tensor=input_img.unsqueeze(0),
                input_tensor = input_img,
                #targets=[ClassifierOutputTarget(l.item()) for l in self.label],
                targets=None)#(128, 32, 32)                        
        elif cam_mode == "eigen":    
            with GradCAM(model=fx_model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
                cam.batch_size=train_batch
                grayscale_cam = cam(
                #input_tensor=input_img.unsqueeze(0),
                input_tensor = input_img,
                #targets=[ClassifierOutputTarget(l.item()) for l in self.label],
                targets=None)#(128, 32, 32)   

        fx_model.train()
        rgb_img = torch.squeeze(rgb_img)
    
        grayscale_cam = np.array(grayscale_cam,dtype=np.float32)
        grayscale_cam = torch.from_numpy(grayscale_cam)
        rgb_img = torch.permute(rgb_img,(1,0,2,3))
        rgb_img = rgb_img.to("cuda")
        grayscale_cam = grayscale_cam.to("cuda")
        
        rgb_img = grayscale_cam*rgb_img  
                    
        rgb_img = torch.permute(rgb_img,(1,0,2,3))
        rgb_imgs = rgb_img.to(device)
        
        optimizer.zero_grad()
        c_outputs = fx_model(rgb_imgs,"g_normal")
        #c_outputs = fx_model(rgb_imgs)
        c_loss = criterion(c_outputs, c_labels)
    
        model_loss = loss + c_loss    
        c_loss.backward()
        
        #model_loss.backward()
        running_loss += c_loss.item()
        pred = torch.argmax(c_outputs, dim=1)
        running_acc += torch.mean(pred.eq(c_labels).float())
        optimizer.step()
    #imshow(torchvision.utils.make_grid(rgb_img),eps)
    running_loss /= len(trainloader)
    running_acc /= len(trainloader)


    #validation
    fx_model.eval()
    val_running_loss = 0.0
    val_g_running_acc = 0.0
    val_n_running_acc = 0.0
    for val_imgs, val_labels in tqdm(val_load):
        val_c_imgs = val_imgs
        val_c_labels = val_labels

       
        
        norm = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_imgs = norm(val_imgs)     #<torch.float32>
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = fx_model(val_imgs,"g_normal")
        val_loss = criterion(val_outputs, val_labels)
        
        val_running_loss += val_loss.item()
        val_pred = torch.argmax(val_outputs, dim=1)
        val_n_running_acc += torch.mean(val_pred.eq(val_labels).float())

        
        #cam_img forward
        val_c_labels = val_c_labels.to(device)
        norm = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_input_img = norm(val_c_imgs)     #<torch.float32>
        val_rgb_img = val_c_imgs 
            
        
                
        target_layers = [
            getattr(fx_model.backbone.layer1, '2'), 
            getattr(fx_model.backbone.layer2, '3'), 
            getattr(fx_model.backbone.layer4, '2'),
            getattr(fx_model.backbone.layer3, '5')]
        
        
        '''
        target_layers = [
            fx_model.backbone.layer1[-1],
            fx_model.backbone.layer2[-1],
            fx_model.backbone.layer3[-1],
            fx_model.backbone.layer4[-1]
            ]  '''
                      
        #input_img = input_img[0].to("cuda")
        val_input_img = val_input_img.to("cuda")
        #input_img = input_img.unsqueeze(0) 
       

        if cam_mode == "grad_cam":
            with GradCAM(model=fx_model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
                cam.batch_size=val_batch
                val_grayscale_cam = cam(
                #input_tensor=input_img.unsqueeze(0),
                input_tensor = val_input_img,
                #targets=[ClassifierOutputTarget(l.item()) for l in self.label],
                targets=None)#(128, 32, 32)                        
        elif cam_mode == "eigen":    
            with GradCAM(model=fx_model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
                cam.batch_size=val_batch
                val_grayscale_cam = cam(
                #input_tensor=input_img.unsqueeze(0),
                input_tensor = val_input_img,
                #targets=[ClassifierOutputTarget(l.item()) for l in self.label],
                targets=None)#(128, 32, 32)   

        
        val_rgb_img = torch.squeeze(val_rgb_img)
    
        val_grayscale_cam = np.array(val_grayscale_cam,dtype=np.float32)
        val_grayscale_cam = torch.from_numpy(val_grayscale_cam)
        val_rgb_img = torch.permute(val_rgb_img,(1,0,2,3))
        val_rgb_img = val_rgb_img.to("cuda")
        val_grayscale_cam = val_grayscale_cam.to("cuda")
        
        val_rgb_img = val_grayscale_cam*val_rgb_img  
                    
        val_rgb_img = torch.permute(val_rgb_img,(1,0,2,3))
        val_rgb_imgs = val_rgb_img.to(device)
   
        
        val_c_outputs = fx_model(val_rgb_imgs,"g_normal")
        val_c_loss = criterion(val_c_outputs, val_c_labels)
            
        val_running_loss += val_c_loss.item()
        val_pred = torch.argmax(val_c_outputs, dim=1)
        val_g_running_acc += torch.mean(val_pred.eq(val_c_labels).float())
    imshow(torchvision.utils.make_grid(val_rgb_imgs),eps)
    
    val_running_loss /= len(val_load)
    val_n_running_acc /= len(val_load)
    val_g_running_acc /= len(val_load)

    print(val_n_running_acc,val_g_running_acc)
    print("epoch: {}, loss: {}, acc: {}, \
            val loss: {}, val acc: {}".format(epoch, running_loss/2, running_acc/2, val_running_loss/2, (val_g_running_acc+val_n_running_acc)/2))
                
        


