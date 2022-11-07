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

#data aug
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

#data load
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")



#Resnet model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
model = model.to(device)

optimizer=optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

print("****************************************************")
print(device)
num_epochs = 20
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

#gcam_visiualize

model.eval()
#model = nn.DataParallel(model)
target_layers = [model.layer4[-1]]
#targets = [ClassifierOutputTarget(10)]*64
i=0



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
    



for images,labels in tqdm(validation_dataloader):
    i=i+1
    if i>5:
        break


    
    #img = images.to('cpu').detach().numpy().copy()
    print(images)
    imshow(torchvision.utils.make_grid(images),str(i))
    '''
    visual=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_NO_gcam'+str(i)+dt_now_str+'.jpg',visual)
    #labels = labels.to(device)
    '''

    images = images.to(device)
    cam = GradCAM(model=model,target_layers=target_layers,use_cuda=True) 
       
    input_tensor = images
          
    rgb_img = images.to('cpu').detach().numpy().copy()
    rgb_img = np.float32(rgb_img)
    print("#############################################")
    print(input_tensor)
          
    grayscale_cam = cam(input_tensor=input_tensor,targets=None,aug_smooth=True)*255
    print("*********************************",rgb_img)
                    
    print("grays:",grayscale_cam.shape)
    print("RGB:",rgb_img.shape)
    rgb_img = np.transpose(rgb_img[0],(1,2,0))
                
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0,:], use_rgb=True)
    visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_gcam'+str(i)+dt_now_str+'.jpg',visualization)

print("000000000000000000000000000000000000000000000")

visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
cv2.imwrite('./result/cifar10_gcam'+dt_now_str+'.jpg',visualization)
print(visualization)



'''        
dataiter = iter(train_dataloader)
images, labels = dataiter.next() 
input_tensor = images.to(device)
rgb_img = images.permute(0,2,3,1).numpy()  
cam = GradCAM(model=model,target_layers=target_layers,use_cuda=True) 

grayscale_cam = cam(input_tensor=input_tensor,targets=None) 
visualization = show_cam_on_image(rgb_img, grayscale_cam[0,:], use_rgb=True)
'''
