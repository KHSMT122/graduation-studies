from ast import arg
from email.mime import image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms,models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import datetime
import pytz

import os
#pip install grad-cam
#$ pip install opencv-python
import cv2


#gcam import 
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image

#device CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#datetime
dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
dt_now_str = str(dt_now)
dt_now=dt_now.strftime('/%m_%d_%H:%M')

#data section
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
validation_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

#Resnet model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
model.load_state_dict(torch.load("./model/model.pth"))
model = model.to(device)

'''model_path = "./model/10_29_11:26.pth"
model.load_state_dict(torch.load(model_path))
'''

#gcam_visiualize
model.eval()
target_layers = [model.layer3[-1]]

ds = datasets.CIFAR10(root="./data", train=False, download=True)

def imshow(img,num):
    # 非正規化する
    img = img / 2 + 0.5
    # torch.Tensor型からnumpy.ndarray型に変換する
    #print(type(img)) # <class 'torch.Tensor'>
    npimg = img.numpy()
    print(type(npimg))    
    # 形状を（RGB、縦、横）から（縦、横、RGB）に変換する
    print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    #npimg.save('./result/cifar10_vis'+dt_now_str+'.jpg',npimg*255)
    
    print(npimg)
    npimg=cv2.cvtColor(npimg,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_NO_CAM_'+num+dt_now_str+'.jpg',npimg*255)

    
i=0
for i in tqdm(range(10)):
    
    rgb_img,label = ds[i]
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img,mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
    imshow(torchvision.utils.make_grid(input_tensor),str(i))
    

    input_tensor = input_tensor.to(device)
    cam = GradCAM(model=model,target_layers=target_layers,use_cuda=True) 
       
    
    print(input_tensor)
    print("#############################################")      
    
    grayscale_cam = cam(input_tensor=input_tensor,targets=None,aug_smooth=True)
    print("*********************************",rgb_img)
                    
    print("grays:",grayscale_cam.shape)
    print("RGB:",rgb_img.shape)
    #rgb_img = np.transpose(rgb_img,(1,2,0))
                
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0,:], use_rgb=True)
    visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_gcam'+str(i)+dt_now_str+'.jpg',visualization)

'''
visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
cv2.imwrite('./result/cifar10_gcam'+dt_now_str+'.jpg',visualization)
print(visualization)
'''

