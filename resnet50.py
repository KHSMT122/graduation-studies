import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms,models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline
import datetime
import pytz

import os



#tensorboard pop install tensorboard 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./logs")

#device CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#PATH
os.makedirs('model',exist_ok=True)
model_PATH = './model'

#datetime
dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

dt_now_str = str(dt_now)
dt_now=dt_now.strftime('/%m_%d_%H:%M')

#data aug
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])

#data load
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
validation_dataloader = DataLoader(val_data, batch_size=128, shuffle=False)
names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")



#Resnet model
model = models.resnet50(pretrained = True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
model = model.to(device)



#early stopping

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

class accEarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = 0   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score > self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation acc increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

#optimizer & loss
#optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.00005)
#optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

criterion = nn.CrossEntropyLoss()
'''
#scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.1
)
'''
#running
print("****************************************************")
print(device)
num_epochs = 200
losses = []
accs = []
val_losses = []
val_accs = []

earlystopping = accEarlyStopping(patience=50, verbose=True)


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
    #
    # validation loop
    #
    model.eval()

    val_running_loss = 0.0
    val_running_acc = 0.0
    for val_imgs, val_labels in tqdm(validation_dataloader):
        
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        
        val_output = model(val_imgs)
        val_loss = criterion(val_output, val_labels)
        val_running_loss += val_loss.item()
        val_pred = torch.argmax(val_output, dim=1)
        val_running_acc += torch.mean(val_pred.eq(val_labels).float())
    
    val_running_loss /= len(validation_dataloader)
    val_running_acc /= len(validation_dataloader)
   
    val_losses.append(val_running_loss)
    val_accs.append(val_running_acc)
    
    writer.add_scalar("running_loss", running_loss, epoch)
    writer.add_scalar("val_running_loss", val_running_loss, epoch)
    writer.add_scalar("val_running_acc", val_running_acc, epoch)
    writer.add_scalar("running_acc", running_acc, epoch)



    print("epoch: {}, loss: {}, acc: {}, \
     val loss: {}, val acc: {}".format(epoch, running_loss, running_acc, val_running_loss, val_running_acc))
     #★毎エポックearlystoppingの判定をさせる★
    earlystopping((val_running_acc), model) #callメソッド呼び出し
    if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
        print("Early Stopping!")
        break
print("save model")
torch.save(model.state_dict(),model_PATH+dt_now+"epochs"+str(num_epochs)+".pth")


