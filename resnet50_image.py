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
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image





#device CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

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
rgb_img = cv2.imread('both.png', 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])

#Resnet model
model = models.resnet50(pretrained=True)
model = model.to(device)
'''

'''
#gcam_visiualize
model.eval()
#model = nn.DataParallel(model)
target_layers = [model.layer4[-1]]
#targets = [ClassifierOutputTarget(10)]*64
def imshow(img):
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
    cv2.imwrite('./result/IMAGE_NO_CAM_'+dt_now_str+'.jpg',npimg*255)
    


    


imshow(torchvision.utils.make_grid(input_tensor))

cam = GradCAM(model=model,target_layers=target_layers,use_cuda=True) 
       
grayscale_cam = cam(input_tensor=input_tensor,targets=None,aug_smooth=True)*255
print("*********************************",rgb_img)
                    
print("grays:",grayscale_cam.shape)
print("RGB:",rgb_img.shape)
#rgb_img = np.transpose(rgb_img,(2,0,1))
print("RGB1:",rgb_img.shape)                
'''
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category,aug_smooth=True)

# In this example grayscale_cam has only one image in the batch:
#grayscale_cam = grayscale_cam[0, :]
#visualization = show_cam_on_image(rgb_img, grayscale_cam)
'''
visualization = show_cam_on_image(rgb_img, grayscale_cam[0,:], use_rgb=True)
visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
cv2.imwrite('./result/IMAGE_gcam'+dt_now_str+'.jpg',visualization)
