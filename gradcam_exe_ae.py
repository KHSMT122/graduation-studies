import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from tqdm import tqdm
BICUBIC = InterpolationMode.BICUBIC

import datetime
import pytz

import os
import cv2

#torch attack (URL:https://github.com/Harry24k/adversarial-attacks-pytorch)
#pip install torchattacks
import torchattacks
from torchattacks import PGD,FGSM

#device CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"




#datetime
dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

dt_now_str = str(dt_now)
dt_now=dt_now.strftime('/%m_%d_%H:%M')


def imshow(img,num):
    # 非正規化する
    #img = img / 2 + 0.5
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
    

def get_loaders(batch_size):
    ds = torchvision.datasets.CIFAR10
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    trainset = ds(root='./data', train=True, download=True, transform=transform)
    indices = torch.arange(5000) 
    trainset = Subset(trainset, indices)

    n_samples = len(trainset)
    train_size = int(len(trainset) * 0.9)
    val_size = n_samples - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])


    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                drop_last=False)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                drop_last=False)

    testset = ds(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                drop_last=False)
    return train_loader, val_loader, test_loader



def plot_ds(dataset, row=10, col=1, figsize=(20,10)):
    fig_img, ax_img = plt.subplots(row, col, figsize=figsize, tight_layout=True)
    plt.figure()
    for i in range(row):
        img1,_ = dataset[i]
        img1 = denormalization(img1)
        img1 = np.squeeze(img1)
        ax_img[i].imshow(img1)
        
    fig_img.savefig("./result/data_sample.png", dpi=100)
    plt.close()

def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def denormalization(x):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    x = inverse_normalize(x, mean, std)
    x = x.cpu().detach().numpy()
    # x = (x.transpose(1, 2, 0)).astype(np.uint8)
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)

    return x

train_loader, val_loader, test_loader = get_loaders(batch_size=32)
plot_ds(train_loader.dataset)

class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

'''
class ClassifierModel(torch.nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()

        # 畳み込み層
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=128,
                      kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(in_channels=128, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.Flatten()
        )
        # 全結合層
        self.mlp_layers = torch.nn.Sequential(
            torch.nn.Linear(256, 50),
            torch.nn.ReLU(True),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        y = self.mlp_layers(x)
        return y

'''
# 学習など
model = models.resnet50(pretrained = True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

#model = ClassifierModel()
model = model.to("cuda")
epochs = 20
train_loss = AverageMeter("train_loss")
train_acc = AverageMeter("train_acc")
val_loss = AverageMeter("val_loss")
val_acc = AverageMeter("val_acc")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
'''
min_loss = np.inf
for epoch in range(epochs):
    model.train()
    for x, y in tqdm(train_loader):
        x = x.to("cuda")
        y = y.to("cuda")
        optimizer.zero_grad()
        outputs = model(x)
        
        loss = criterion(outputs.squeeze(), y)
        
        _, predicted = torch.max(outputs.data, 1)
    
        accuracy = (predicted==y).sum().item()/y.size(0)
        

        train_loss.update(loss.data)
        train_acc.update(accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    for x, y in tqdm(val_loader):
        x = x.to("cuda")
        y = y.to("cuda")
        outputs = model(x)
        loss = criterion(outputs.squeeze(), y)
        val_loss.update(loss.data)

        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted==y).sum().item()/y.size(0)
        val_acc.update(accuracy)

    if val_loss.avg < min_loss:
        print("save model")
        torch.save(model.state_dict(), "./model/model.pth")
        min_loss = val_loss.avg

    print(
        "[epoch :{:.1f} train_loss: {} val_loss: {} train_acc: {} val_acc: {}] ".format(
            epoch, train_loss.avg, val_loss.avg,
            train_acc.avg, val_acc.avg
        )
    )
    scheduler.step()
    train_loss.reset()
    val_loss.reset()
'''
model.load_state_dict(torch.load("./model/model.pth"))

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM,EigenCAM
model.eval()

atk = FGSM(model,eps=0/255)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


input_transform = transforms.Compose([
    transforms.Resize(32, interpolation=BICUBIC),
    #transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_transform = transforms.Compose([
    transforms.Resize(32, interpolation=BICUBIC),
    #transforms.ToTensor(),
])
set_transforms = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=set_transforms)
trainset_data = DataLoader(trainset,batch_size=1,shuffle=False)
for i in range(2):
    '''
    set_iter = iter(trainset_data)
    img, label = set_iter.next()
    
    input_img = input_transform(img)
    
    img = img.to(device)
    label = label.to(device)
    '''
    
    img,label = trainset[i]

    img = img.to(device) 
    label = torch.tensor([label],dtype=torch.int64)
    
    label = label.to(device)

    img = atk(img,label)
    
    '''
    img = img.unsqueeze_(0)
    label = torch.tensor(label)
    label = label.unsqueeze_(0)
    '''
    input_img = input_transform(img)
    rgb_img = img_transform(img)
    
    
    target_layers = [model.layer3[-1],model.layer4[-1],model.layer2[-1],model.layer1[-1]]
    cam = GradCAM(
        model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
    )
   
    input_img = input_img.squeeze(0)
    
    grayscale_cam = cam(
        input_tensor=input_img.unsqueeze(0),
        #targets=[ClassifierOutputTarget(label)],
        targets=None
    )
    
    grayscale_cam = grayscale_cam[0, :]
    
    '''
   img.detach().cpu().numpy()
    grayscale_cam.detach().cpu().numpy()
    '''
    
    
    
    rgb_img = torch.squeeze(rgb_img)  # <torch.Size([3, 32, 32])>
   
    
    #####visualizationi
    
    grayscale_cam = np.array(grayscale_cam,dtype=np.float32)
    #grayscale_cam = np.ndarray(grayscale_cam)
    
    print("*******************show******************************")
    grayscale_cam = torch.from_numpy(grayscale_cam)
    print(rgb_img.dtype)
    
    print(grayscale_cam.dtype)
    print("*******************show******************************")

    
    visualization = show_cam_on_image(rgb_img.permute(1,2,0), grayscale_cam, use_rgb=True)
    
    '''
        #visualization = show_cam_on_image(input_img.permute(1, 2, 0), grayscale_cam, use_rgb=True)
        fig, ax = plt.subplots(1,2)    
        ax[0].imshow(img.permute(1, 2, 0).numpy())
        ax[1].imshow(visualization)

    '''
    
    imshow(torchvision.utils.make_grid(img),str(i))
    visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result/cifar10_gcam'+str(i)+dt_now_str+'.jpg',visualization)
    
