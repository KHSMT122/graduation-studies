from torch.utils.data import dataset
from torchvision import models,datasets, transforms
from transform import grad_transforms_train
from transform import grad_transforms_val
import torch
import torch.nn as nn
from tqdm import tqdm

class MyDataset_train():
    def __init__(self,model_file,model_name,cam_mode):
        super().__init__()
        self.model_file = model_file
        self.model_name = model_name
        self.rs_train_img = []
        self.rs_train_label = []
        self.rs_train_data = []
        self.num = 0
        self.cam_mode = cam_mode
        #self.train_data = datasets.CIFAR10(root="./data", train=True, download=True)
        self.train_data = datasets.STL10(root='./pytorch-cifar-master/data',split='train',download=True)


    def __call__(self):
        
        print("train_data_cifar10")
        print("cam method:",self.cam_mode)

        if self.model_name == "resnet50":
            
            self.model = models.resnet50(pretrained = True)
            self.model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
            self.model.load_state_dict(torch.load("./pytorch-cifar-master/checkpoint/"+self.model_file+".pth"))

        for train in tqdm(self.train_data):
            '''
            if self.num>=1000:
                break
            '''
            train_img,train_label=train 
            train_transform = grad_transforms_train(train_img,train_label,self.model,self.cam_mode)   
            s_train_img,s_train_label = train_transform()
        
            s_train_img = s_train_img.squeeze()
                
            #print(s_train_img*255)            
            #self.rs_train_img.append(s_train_img)
            #self.rs_train_label.append(s_train_label)
            #self.rs_train_img = torch.tensor(s_train_img,dtype=torch.float32)
            #self.rs_train_label = torch.tensor(s_train_label,dtype=torch.int64)
            self.rs_train_data.append([s_train_img,s_train_label])
            self.num = self.num+1
        return self.rs_train_data 
    def __len__(self):
        return len(self.rs_train_img)



            
            
class MyDataset_val():
    def __init__(self,model_name,atk_name,model_file,eps,alpha,step,cam_mode):
        super().__init__()
        self.model_file = model_file
        self.model_name = model_name
        self.atk_name = atk_name
        self.rs_val_img = []
        self.rs_val_label = []
        self.eps = eps
        self.alpha = alpha
        self.step = step
        #self.val_data = datasets.CIFAR10(root="./data", train=False, download=True)


        self.val_data = datasets.STL10(root='./pytorch-cifar-master/data',split='test',download=True)

        self.num = 0
        self.cam_mode = cam_mode
        self.rs_val_data = []
    def __call__(self):
        print("val_data_cifar10")
        if self.model_name == "resnet50":
            self.model = models.resnet50(pretrained = True)
            self.model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
            self.model.load_state_dict(torch.load("./pytorch-cifar-master/checkpoint/"+self.model_file+".pth"))
            print(self.model_name,"was loaded")
        print("atack:",self.atk_name)
        print("cam method:",self.cam_mode)
    
        for val in tqdm(self.val_data):
            '''              
            if self.num>=500:
                break
            '''
            s_val_img = 0
            s_val_label= 0
            val_img,val_label = val
            val_transform = grad_transforms_val(val_img,val_label,self.model,self.atk_name,self.eps,self.alpha,self.step,self.cam_mode)
            s_val_img,s_val_label = val_transform()

            
            self.rs_val_data.append([s_val_img,s_val_label])

            #self.rs_val_img.append(s_val_img)
            #self.rs_val_label.append(s_val_label)
            self.num = self.num +1 
            
        #self.rs_val_data = torch.tensor(self.rs_val_data)
        return self.rs_val_data
    def __len__(self):
        return len(self.rs_val_img)    