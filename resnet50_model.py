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
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = create_feature_extractor(self.backbone, ["flatten"])
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 10)
        )
    def forward(self, x, change_head: str):
        x = self.backbone(x)["flatten"]
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


fx_model = models.resnet50(pretrained = True)
fx_model.fc = nn.Linear(2048,10)
#print(fx_model)
#fx_model = Network()
fx_model = fx_model.to("cuda")

#print(fx_model.backbone.layer2)
#print(getattr(fx_model.backbone.layer2, '0'))
optimizer = optim.SGD(
        fx_model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.00005
        )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()


atk = PGD(fx_model,eps=4/255,alpha = 8/255,steps=2, random_start=False)
atk = FGSM(fx_model,eps=2/255)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])




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
cam_mode = "eigen"
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
        outputs = fx_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        fx_model.eval()

        



        c_labels = c_labels.to(device)
        norm = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_img = norm(c_imgs)     #<torch.float32>
        rgb_img = c_imgs 
            
        fx_model.eval()
        
        #target_layers = [getattr(fx_model.backbone.layer1, '2'), getattr(fx_model.backbone.layer2, '3'), getattr(fx_model.backbone.layer4, '2'),getattr(fx_model.backbone.layer3, '5')]
        
        print("*"*80)
        #print(target_layers)
        #target_layers = [fx_model.backbone.module.layer2[-1],fx_model.backbone.module.layer2[-1],fx_model.backbone.module.layer3[-1]]
        target_layers = [fx_model.layer3[-1],fx_model.layer4[-1],fx_model.layer2[-1],fx_model.layer1[-1]]
        
        #input_img = input_img[0].to("cuda")
        input_img = input_img.to("cuda")
        #input_img = input_img.unsqueeze(0) 
       
        print(input_img.shape)
        '''


        cam = GradCAM(
            model=fx_model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
        )
        cam.batch_size = 1
        #input_img = input_img.squeeze(0)
        print(input_img.shape)
        grayscale_cam = cam(
            input_tensor=input_img,
            #targets=[ClassifierOutputTarget(label)],
            targets=None
        )
        
        
        
        '''
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
        #cam_imgs = cam_imgs.to('cpu').detach().numpy().copy()
        #ims(torchvision.utils.make_grid(cam_imgs),128)     
        optimizer.zero_grad()
        c_outputs = fx_model(rgb_imgs)
        loss = criterion(c_outputs, c_labels)
            
        loss.backward()
        optimizer.step()

        


    