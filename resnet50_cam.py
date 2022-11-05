import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms,models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


import datetime
import pytz
import cv2

import os



#torch attack (URL:https://github.com/Harry24k/adversarial-attacks-pytorch)
#pip install torchattacks
import torchattacks
from torchattacks import PGD,FGSM



#device CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#PATH
os.makedirs('model',exist_ok=True)
model_PATH = './model'

#datetime
dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

dt_now_str = str(dt_now)
dt_now=dt_now.strftime('/%m/%d_%H:%M:%S')
#data aug
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

#data load
batch_size=4
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
train_load = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_load = DataLoader(val_data, batch_size=batch_size, shuffle=False)


def imshow(img,num):
    # 非正規化する
    img = img / 2 + 0.5
    # torch.Tensor型からnumpy.ndarray型に変換する
    print(type(img)) # <class 'torch.Tensor'>
    npimg = img.numpy()
    print(type(npimg))    
    # 形状を（RGB、縦、横）から（縦、横、RGB）に変換する
    print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    #npimg.save('./result/cifar10_vis'+dt_now_str+'.jpg',npimg*255)
    
    print(npimg)
    npimg=cv2.cvtColor(npimg,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_'+dt_now_str+'.jpg',npimg*255)
    
    #cv2.imwrite('./result/cifar10_vis2.jpg',npimg*255)

def ad_imshow(a_img,num):
    # 非正規化する
    a_img = a_img / 2 + 0.5
    # torch.Tensor型からnumpy.ndarray型に変換する
    print(type(a_img)) # <class 'torch.Tensor'>
    #npimg = img.numpy()
    a_npimg = a_img.to('cpu').detach().numpy().copy()
    print(type(a_npimg)) 

    print(a_npimg.shape)
    #npimg = img.numpy()
    print(type(a_npimg))    
    # 形状を（RGB、縦、横）から（縦、横、RGB）に変換する
    print(a_npimg.shape)
    a_npimg = np.transpose(a_npimg, (1, 2, 0))
    print(a_npimg.shape)
    
    print(a_npimg)
    #a_npimg.save('./result/cifar10_vis_FGSM'+dt_now_str+'.jpg',a_npimg*255)
    
    a_npimg=cv2.cvtColor(a_npimg,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_vis_FGSM'+dt_now_str+'.jpg',a_npimg*255)

    
def checkimage(img):
    print(img.dtype)
    print(img.shape)
    

#show imgs
dataiter = iter(train_load)
images, labels = dataiter.next()
'''
checkimage(images)
print("111111111111111111111111111111111111111111")
'''
print("888888888888888888888888888888888888888888")
print(images)


imshow(torchvision.utils.make_grid(images),batch_size)
#imshow(images),batch_size)
print(' '.join('%5s' % names[labels[j]] for j in range(4)))
print("success_cifar10_vis"+dt_now_str+".jpg")

#Resnet model
model = models.resnet50(pretrained = True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
#model = model.to(device)

#optimizer & loss
optimizer=optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

#attack
atk = PGD(model, eps=32/255, alpha=2/225, steps=10, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
atk_img = atk(images,labels)


#ad_imshow(atk_img,batch_size)
print("999999999999999999999999999999999999999")
print(atk_img)
ad_imshow(torchvision.utils.make_grid(atk_img),batch_size)
print(' '.join('%5s' % names[labels[j]] for j in range(4)))
print("success_cifar10_vis_FGSM"+dt_now_str+".jpg")
