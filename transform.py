import torch
import numpy as np
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import cv2

import torchattacks
from torchattacks import PGD,FGSM,CW


from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM,EigenCAM


class grad_transforms_train():
    def __init__(self,image,label,model,cam_mode):
        self.image = image
        self.label = label
        self.model = model
        self.conut = 0
        self.cam_mode = cam_mode


    def __call__(self):
       

        self.model.eval()

        input_transform= transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_transform = transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),

        ])       
        self.label = torch.tensor([self.label],dtype=torch.int64)
        
        input_img = input_transform(self.image)     #<torch.float32>
        rgb_img = img_transform(self.image)                #<torch.float32>

        target_layers = [self.model.layer3[-1],self.model.layer4[-1],self.model.layer2[-1],self.model.layer1[-1]]
        if self.cam_mode == "eigen":
            cam = EigenCAM(
                model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
            )
            
        elif self.cam_mode == "grad_cam":
            cam = GradCAM(
                model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
            )
            
        
        grayscale_cam = cam(
            input_tensor=input_img.unsqueeze(0),
            #targets=[ClassifierOutputTarget(label)],
            targets=None
        )
        
        #Saliency : true  Noise : false 
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = torch.squeeze(rgb_img)
        grayscale_cam = np.array(grayscale_cam,dtype=np.float32)#(32,32)
        rgb_img = rgb_img.to('cpu').detach().numpy().copy()#(3,32,32)
        visualization = show_cam_on_image(rgb_img.transpose(1,2,0), grayscale_cam, use_rgb=True)
        
    
        rgb_img = rgb_img * grayscale_cam 
        
        
        rgb_img = torch.tensor([rgb_img],dtype=torch.float32)

        #KOKOMADE okS
        return rgb_img,self.label
    

class grad_transforms_val():
    def __init__(self,image,label,model,atk_name,eps,alpha,step,cam_mode):
        
        self.image = image
        self.label = label
        self.conut = 0
        self.atk_name = atk_name
        self.eps = eps
        self.alpha = alpha
        self.step = step
        self.model = model
        self.cam_mode = cam_mode
        
        #CW attack
        self.lr = 0.01
        self.kappa = 0

    def __call__(self):
        
        if self.atk_name == "PGD":
            atk = PGD(self.model,eps=self.eps,alpha=self.alpha,steps=self.step)
            atk.set_return_type(type='float')
            atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
            

        elif self.atk_name == "FGSM":
            atk = FGSM(self.model,eps=self.eps)
            atk.set_return_type(type='float')
            atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
            
        elif self.atk_name == "CW":
            atk = CW(self.model, c=0.001, kappa=0, steps=1000, lr=0.01)
            atk.set_return_type(type='float')
            atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
            

        self.model.eval()

        input_transform= transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_transform = transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            #transforms.ToTensor(),
        ])       
        '''set_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])'''

    
        self.image = np.array(self.image)

        self.image = (self.image/255).astype(np.float32)
       

        image = torch.tensor([self.image],dtype=torch.float32)
        image = image.to("cuda")
        label = torch.tensor([self.label],dtype=torch.int64)
        label = label.to("cuda")
        image = image.permute(0,3,1,2)
        image = image.squeeze()
        image = atk(image,label)
    
        input_img = input_transform(image)     #<torch.float32>
        rgb_img = img_transform(image)                #<torch.float32>
        

        target_layers = [self.model.layer3[-1],self.model.layer4[-1],self.model.layer2[-1],self.model.layer1[-1]]
        
        if self.cam_mode == "eigen":
            cam = EigenCAM(
                model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
            )

        elif self.cam_mode == "grad_cam":
            cam = GradCAM(
                model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
            )      
            
        input_img = input_img.squeeze(0)

        grayscale_cam = cam(
            input_tensor=input_img.unsqueeze(0),
            #targets=[ClassifierOutputTarget(label)],
            targets=None
        )
    
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = torch.squeeze(rgb_img)
        grayscale_cam = np.array(grayscale_cam,dtype=np.float32)
        rgb_img = rgb_img.to('cpu').detach().numpy().copy()
        #visualization = show_cam_on_image(rgb_img.transpose(1,2,0), grayscale_cam, use_rgb=True)
        rgb_img = rgb_img * grayscale_cam # rgb_img:float32

        rgb_img = torch.from_numpy(rgb_img.astype(np.float32)).clone()# rgb_img:torch float32
        self.label = torch.tensor([self.label],dtype=torch.int64)
        return rgb_img,self.label  


        #rgb_img = np.transpose(rgb_img,(1,2,0))
        '''
        visualization = show_cam_on_image(img.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
        visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
        '''

        




