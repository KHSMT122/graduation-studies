from torch.utils.data import dataset
from torchvision import models,datasets, transforms
from transform import grad_transforms_train
from transform import grad_transforms_val
import torch
from tqdm import tqdm

class MyDataset_train():
    def __init__(self,model_file,model_name):
        super().__init__()
        self.model_path = model_file
        self.model_name = model_name
        self.rs_train_img = []
        self.rs_train_label = []
        self.train_data = datasets.CIFAR10(root="./data", train=True, download=True)


    def __call__(self):
        
        print("train_data_cifar10")
        for train_img,train_label in tqdm((self.train_data)):
            print(train_img)
            train_transform = grad_transforms_train(train_img,train_label,self.model_path,self.model_name)   
            s_train_img,s_train_label = train_transform()            
            self.rs_train_img.append(s_train_img)
            self.rs_train_label.append(s_train_label)
        return self.rs_train_img,self.rs_train_label   
    def __len__(self):
        return len(self.rs_train_img)



            
            
class MyDataset_val():
    def __init__(self,model_name,atk_name,model_file,eps,alpha,step):
        super().__init__()
        self.model_path = model_file
        self.model_name = model_name
        self.atk_name = atk_name
        self.rs_val_img = []
        self.rs_val_label = []
        self.eps = eps
        self.alpha = alpha
        self.step = step
        self.val_data = datasets.CIFAR10(root="./data", train=False, download=True)
    def __call__(self):
        print("val_data_cifar10")
    
        for val_img,val_label in tqdm((self.val_data)):
            
            val_transform = grad_transforms_val(val_img,val_label,self.model_name,self.atk_name,self.model_path,self.eps,self.alpha,self.step)
            s_val_img,s_val_label = val_transform()
            self.rs_val_img.append(s_val_img)
            self.rs_val_label.append(s_val_label)
        return self.rs_val_img,self.rs_val_label
    def __len__(self):
        return len(self.rs_val_img)    