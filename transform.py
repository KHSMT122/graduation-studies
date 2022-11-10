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
    def __init__(self,image,label,model_file,model_name):
        self.model_path = model_file
        self.image = image
        self.label = label
        self.model_name = model_name
        self.conut = 0


    def __call__(self):
        if self.model_name == "resnet50":
            
            model = models.resnet50(pretrained = True)
            model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
            model.load_state_dict(torch.load("./model/"+self.model_path+".pth"))


        model.eval()

        input_transform= transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_transform = transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),

        ])       

        
        input_img = input_transform(self.image)     #<torch.float32>
        rgb_img = img_transform(self.image)                #<torch.float32>

        target_layers = [model.layer3[-1],model.layer4[-1],model.layer2[-1],model.layer1[-1]]
        cam = EigenCAM(
            model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
        )
        
        grayscale_cam = cam(
            input_tensor=input_img.unsqueeze(0),
            #targets=[ClassifierOutputTarget(label)],
            targets=None
        )
        #Saliency : true  Noise : false 
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = torch.squeeze(rgb_img)
        grayscale_cam = np.array(grayscale_cam,dtype=np.float32)
        rgb_img = rgb_img.to('cpu').detach().numpy().copy()
        visualization = show_cam_on_image(rgb_img.transpose(1,2,0), grayscale_cam, use_rgb=True)
        rgb_img = rgb_img * grayscale_cam 
        #KOKOMADE okS
        return rgb_img,self.label
    

class grad_transforms_val():
    def __init__(self,image,label,model_name,atk_name,model_file,eps,alpha,step):
        
        self.model_file = model_file
        self.image = image
        self.label = label
        self.conut = 0
        self.atk_name = atk_name
        self.model_name = model_name
        self.eps = eps
        self.alpha = alpha
        self.step = step
        
        #CW attack
        self.lr = 0.01
        self.kappa = 0
        


    def __call__(self):
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained = True)
            model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
            model.load_state_dict(torch.load("./model/"+self.model_file+".pth"))
            print(self.model_name,"was loaded")

            


        if self.atk_name == "PGD":
            atk = PGD(model,eps=self.eps,alpha=self.alpha,step=self.step)
            atk.set_return_type(type='float')
            atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
            print("atack:",self.atk_name)

        elif self.atk_name == "FGSM":
            atk = FGSM(model,eps=self.eps)
            atk.set_return_type(type='float')
            atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
            print("atack:",self.atk_name)
        elif self.atk_name == "CW":
            atk = CW(model,c=1,eps=self.eps,)
            atk.set_return_type(type='float')
            atk.set_normalization_used(mean=[0, 0, 0], std=[1, 1, 1])
            print("atack:",self.atk_name)

        model.eval()

        input_transform= transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_transform = transforms.Compose([
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),
        ])       
        '''set_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])'''
        
        print("********************************************************",self.image)
        image = torch.tensor(self.image,dtype=torch.float32)
        image = image.to("cuda")
        label = torch.tensor([self.label],dtype=torch.int64)
        label = self.label.to("cuda")

        image = atk(self.image,self.label)

        input_img = input_transform(image)     #<torch.float32>
        rgb_img = img_transform(image)                #<torch.float32>

        target_layers = [model.layer3[-1],model.layer4[-1],model.layer2[-1],model.layer1[-1]]
        cam = EigenCAM(
            model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
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
        visualization = show_cam_on_image(rgb_img.transpose(1,2,0), grayscale_cam, use_rgb=True)
        rgb_img = rgb_img * grayscale_cam 
        
        return rgb_img,self.label


        #rgb_img = np.transpose(rgb_img,(1,2,0))
        '''
        visualization = show_cam_on_image(img.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
        visualization=cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
        '''

        




