#!pip install git+https://github.com/okankop/vidaug --quiet

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math
import albumentations as A
from matplotlib.patches import Circle
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR,StepLR, ReduceLROnPlateau
from scipy.ndimage.morphology import grey_dilation
from statistics import mean, median, mode
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import torchvision.models as models
import random
import pandas as pd
from sklearn.preprocessing import normalize,MinMaxScaler,StandardScaler

from vidaug import augmentors as va

als = 1
hc = 1
ps = 1
num_classes = hc+als+ps

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(device)

class VideoDataset_patches(Dataset):

    def __init__(self,X,Y,handpicked_features,flag=0):
        self.X = X
        self.Y = Y
        self.flag = flag
        self.always = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 100% probability

        self.seq2 = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            self.always(va.RandomRotate(degrees=30)),
            self.always(va.Salt()),
            self.always(va.Pepper()),
#             self.always(va.ElasticTransformation(0.9,0.2))       
        ])
        
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self,idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.flag==1:
            x = x.cpu().detach().numpy()
            final_vid = np.array(self.seq2(x))  
            final_vid = T.from_numpy(final_vid) 
        else:
            final_vid = x
        del x
        return final_vid,y

X_ALS_patches = np.load('../input/mouth-patches-w-handpicked/16_len_als_handpicked/X_ALS_pch.npy')
X_PS_patches = np.load('../input/mouth-patches-w-handpicked/16_len_ps_handpicked/X_PS_pch.npy')
X_HC_patches = np.load('../input/mouth-patches-w-handpicked/16_len_hc_handpicked/X_HC_pch.npy')

ALS_lbls = np.load('../input/mouth-patches-w-handpicked/16_len_als_handpicked/ALS_lbls.npy')
PS_lbls = np.load('../input/mouth-patches-w-handpicked/16_len_ps_handpicked/PS_lbls.npy')
HC_lbls = np.load('../input/mouth-patches-w-handpicked/16_len_hc_handpicked/HC_lbls.npy')


from torchvision.utils import make_grid
def show_batch(frames):
    images = T.from_numpy(np.array(frames)).permute(0, 3, 1,2)
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=12).permute(1, 2, 0))
#     fig.savefig('./ALS/' + lbl + '/'+ str(k) +'.jpg')


show_batch(X_ALS_patches[0]/255.)

plt.imshow(X_ALS_patches[65][12]/255.)

print(X_ALS_patches.shape)
print(X_PS_patches.shape)
print(X_HC_patches.shape)


als_vids = np.array(['A002','A006','A008','A009','A010','A011','A014','A015','A016'])
hc_vids = np.array(['N001','N002','N003','N004','N007','N008','N010','N011','N012','N017'])
ps_vids = np.array(['S001', 'S002', 'S003', 'S005', 'S006','S007', 'S008', 'S013'])
from sklearn.model_selection import KFold,RepeatedKFold
kf = RepeatedKFold(n_splits=3,n_repeats=3,random_state=0)
print(kf.get_n_splits(als_vids))

als_indexes_train = []
als_indexes_test = []
als_indexes_val = []

print(kf)
for train_index, test_index in kf.split(als_vids):
    print("TRAIN:", train_index, "TEST:", test_index[:-1],"VAL:",test_index[-1:])
    als_indexes_train.append(train_index)
    als_indexes_test.append(test_index[:-1])
    als_indexes_val.append(test_index[-1:])
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

print(kf.get_n_splits(ps_vids))

ps_indexes_train = []
ps_indexes_test = []
ps_indexes_val = []

print(kf)
for train_index, test_index in kf.split(ps_vids):
    print("TRAIN:", train_index, "TEST:", test_index[:-1],"VAL:",test_index[-1:])
    ps_indexes_train.append(train_index)
    ps_indexes_test.append(test_index[:-1])
    ps_indexes_val.append(test_index[-1:])

print(kf.get_n_splits(hc_vids))

hc_indexes_train = []
hc_indexes_test = []
hc_indexes_val = []

print(kf)
for train_index, test_index in kf.split(hc_vids):
    print("TRAIN:", train_index, "TEST:", test_index[:-1],"VAL:",test_index[-1:])
    hc_indexes_train.append(train_index)
    hc_indexes_test.append(test_index[:-1])
    hc_indexes_val.append(test_index[-1:])



def get_indexes(name,roll):
    
    index_train_ALS = []
    index_test_ALS= []
    index_val_ALS=[]

    index_train_PS = []
    index_test_PS= []
    index_val_PS=[]

    index_train_HC = []
    index_test_HC = []
    index_val_HC = []

    for i, pName in enumerate(ALS_lbls):
        if 'NORMAL' in pName:
            if pName.split('_')[0] in als_vids[als_indexes_test[roll]]:
                index_test_ALS.append(i)
            elif pName.split('_')[0] in als_vids[als_indexes_val[roll]]:
                index_val_ALS.append(i)
            elif pName.split('_')[0] in als_vids[als_indexes_train[roll]]:
                index_train_ALS.append(i)

    for i, pName in enumerate(HC_lbls):
        if 'NORMAL' in pName:
            if pName.split('_')[0] in hc_vids[hc_indexes_test[roll]]:
                index_test_HC.append(i)
            elif pName.split('_')[0] in hc_vids[hc_indexes_val[roll]]:
                index_val_HC.append(i)    
            elif pName.split('_')[0] in hc_vids[hc_indexes_train[roll]]:
                index_train_HC.append(i)

    for i, pName in enumerate(PS_lbls):
        if 'NORMAL' in pName:
            if pName.split('_')[0] in ps_vids[ps_indexes_test[roll]]:
                index_test_PS.append(i)
            elif pName.split('_')[0] in ps_vids[ps_indexes_val[roll]]:
                index_val_PS.append(i)
            elif pName.split('_')[0] in ps_vids[ps_indexes_train[roll]]:
                index_train_PS.append(i)
                
                
    if name=='als':
        return index_train_ALS,index_val_ALS,index_test_ALS
    elif name=='ps':
        return index_train_PS,index_val_PS,index_test_PS
    elif name=='hc':
        return index_train_HC,index_val_HC,index_test_HC

def getSplittedData(data, data_lbl, indexes ,handpicked):
    y_train=[]
    y_val=[]
    y_test=[]
    y_test_lbl = []    
    X_train = []
    X_val = []
    X_test = []
    X_train_lbl = []
    X_val_lbl = []
    X_test_lbl = []

    X_train_handpicked = []
    X_val_handpicked = []
    X_test_handpicked = []
    
    
    for j in (indexes[0][0]):
        X_train.append(data[0][j])
        X_train_lbl.append(data_lbl[0][j])
        y_train.append(0)
        X_train_handpicked.append(handpicked[0][j])
        
    for j in (indexes[0][1]):
        X_val.append(data[0][j])
        X_val_lbl.append(data_lbl[0][j])
        y_val.append(0)
        X_val_handpicked.append(handpicked[0][j])
        
    for j in (indexes[0][2]):
        X_test.append(data[0][j])
        X_test_lbl.append(data_lbl[0][j])
        y_test.append(0)
        X_test_handpicked.append(handpicked[0][j])
        
    for j in (indexes[1][0]):
        X_train.append(data[1][j])
        X_train_lbl.append(data_lbl[1][j])
        y_train.append(1)
        X_train_handpicked.append(handpicked[1][j])
        
    for j in (indexes[1][1]):
        X_val.append(data[1][j])
        X_val_lbl.append(data_lbl[1][j])
        y_val.append(1)
        X_val_handpicked.append(handpicked[1][j])
        
    for j in (indexes[1][2]):
        X_test.append(data[1][j])
        X_test_lbl.append(data_lbl[1][j])
        y_test.append(1)
        X_test_handpicked.append(handpicked[1][j])
    
    if len(data)==3:
        for j in (indexes[2][0]):
            X_train.append(data[2][j])
            X_train_lbl.append(data_lbl[2][j])
            y_train.append(2)
            X_train_handpicked.append(handpicked[2][j])
            
        for j in (indexes[2][1]):
            X_val.append(data[2][j])
            X_val_lbl.append(data_lbl[2][j])
            y_val.append(2)
            X_val_handpicked.append(handpicked[2][j])
            
        for j in (indexes[2][2]):
            X_test.append(data[2][j])
            X_test_lbl.append(data_lbl[2][j])
            y_test.append(2)
            X_test_handpicked.append(handpicked[2][j])
    
    return np.array(X_train), np.array(X_val), np.array(X_test), y_train, y_val, y_test, X_train_lbl, X_val_lbl, X_test_lbl, y_test_lbl , np.array(X_train_handpicked),np.array(X_val_handpicked),np.array(X_test_handpicked)

def rolled_DataLoaders(als, hc, ps, roll):

    tmp_ALS = X_ALS_patches
    tmp_PS = X_PS_patches
    tmp_HC = X_HC_patches
    

    tmp_lbl_ALS = ALS_lbls
    tmp_lbl_PS = PS_lbls
    tmp_lbl_HC = HC_lbls
    
    tmp_handpicked_ALS = ALS_handpicked_features
    tmp_handpicked_PS = PS_handpicked_features
    tmp_handpicked_HC = HC_handpicked_features
        
    index_train_ALS,index_val_ALS,index_test_ALS = get_indexes('als',roll)
    index_train_PS,index_val_PS,index_test_PS = get_indexes('ps',roll)
    index_train_HC,index_val_HC,index_test_HC = get_indexes('hc',roll)
    
#     print('Train:')
#     print('ALS:',len(index_train_ALS))
#     print('PS:',len(index_train_PS))
#     print('HC:',len(index_train_HC))
    
#     print('Validation:')
#     print('ALS:',len(index_val_ALS))
#     print('PS:',len(index_val_PS))
#     print('HC:',len(index_val_HC))
    
#     print('Test:')
#     print('ALS:',len(index_test_ALS))
#     print('PS:',len(index_test_PS))
#     print('HC:',len(index_test_HC))
    
    indexes_ALS = [index_train_ALS, index_val_ALS,index_test_ALS]
    indexes_PS = [index_train_PS, index_val_PS,index_test_PS]
    indexes_HC = [index_train_HC, index_val_HC,index_test_HC]
    
#     print(indexes_ALS)
#     print(indexes_PS)
#     print(indexes_HC)
    
    if als ==1 and hc == 1 and ps ==0:
        indexes = [indexes_ALS, indexes_HC]
        data = [tmp_ALS, tmp_HC]
        data_lbl = [tmp_lbl_ALS, tmp_lbl_HC]
        handpicked = [tmp_handpicked_ALS,tmp_handpicked_HC]

                
    elif als ==0 and hc == 1 and ps ==1:
        indexes = [indexes_PS, indexes_HC]
        data = [tmp_PS, tmp_HC]
        data_lbl = [tmp_lbl_PS, tmp_lbl_HC]
        handpicked = [tmp_handpicked_PS,tmp_handpicked_HC]
        
    elif als ==1 and hc == 0 and ps ==1:
        indexes = [indexes_ALS, indexes_PS]
        data = [tmp_ALS, tmp_PS]
        data_lbl = [tmp_lbl_ALS, tmp_lbl_PS]
        handpicked = [tmp_handpicked_ALS,tmp_handpicked_PS]
        
    elif als ==1 and hc == 1 and ps ==1:
        indexes = [indexes_ALS, indexes_PS, indexes_HC]
        data = [tmp_ALS, tmp_PS, tmp_HC]
        data_lbl = [tmp_lbl_ALS, tmp_lbl_PS, tmp_lbl_HC]
        handpicked = [tmp_handpicked_ALS,tmp_handpicked_PS,tmp_handpicked_HC]
    
#####---------------------------------------------------------------------------------------------------------------------------------########

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_lbl, X_val_lbl, X_test_lbl, y_test_lbl ,X_train_handpicked,X_val_handpicked,X_test_handpicked= getSplittedData(data, data_lbl, indexes,handpicked )
    print(X_train.shape)
    print(X_train_handpicked.shape)

#####---------------------------------------------------------------------------------------------------------------------------------########


    y_train = T.Tensor(y_train).type(T.LongTensor).to(device)
    X_train = T.Tensor(X_train).to(device)
    X_train_handpicked = T.Tensor(X_train_handpicked).to(device)

    y_val = T.Tensor(y_val).type(T.LongTensor).to(device)
    X_val = T.Tensor(X_val).to(device)
    X_val_handpicked = T.Tensor(X_val_handpicked).to(device)

    y_test = T.Tensor(y_test).type(T.LongTensor).to(device)
    X_test = T.Tensor(X_test).to(device)
    X_test_handpicked = T.Tensor(X_test_handpicked).to(device)
    
    #print(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])

    train_classification_dataset = VideoDataset_patches(X_train,y_train,X_train_handpicked,flag=1)
    train_classification_dataloader = DataLoader(train_classification_dataset,batch_size=batch_size,shuffle=True)
    val_classification_dataset = VideoDataset_patches(X_val,y_val,X_val_handpicked)
    val_classification_dataloader = DataLoader(val_classification_dataset,batch_size=batch_size,shuffle=False)
    
    
    
    test_classification_dataset = VideoDataset_patches(X_test,y_test,X_test_handpicked)
    test_classification_dataloader = DataLoader(test_classification_dataset,batch_size=1,shuffle=False)
    
    return train_classification_dataloader, val_classification_dataloader, test_classification_dataloader, X_test, y_test , X_train_lbl, X_val_lbl, X_test_lbl, y_test_lbl

batch_size=8



criterion = nn.CrossEntropyLoss()

class DiseaseClassifier(nn.Module):
    
    def training_step(self, batch):
        images, labels,features = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self((images/255.))              # Generate predictions
        loss = criterion(out, labels)   # Calculate loss
        #print((loss*len(images)).detach().item())
        acc = accuracy(out, labels)            # Calculate acc
        #print(acc.item())
        return loss, acc
    
    def validation_step(self, batch):
        images, labels,features = batch 
        images = images.to(device)
        labels = labels.to(device)
        out = self((images/255.))                  # Generate predictions
#         print(out)
#         print('\nOutput -------->',T.argmax(out, dim=1))
#         print('\nlabels ++++++++>',labels)
#         print('\n')
        loss = criterion(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': (loss*len(images)).detach().item(), 'val_acc': acc.item()}
        
    def validation_epoch_end(self, outputs,val_len):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for x in outputs:
            epoch_loss += x['val_loss']
            epoch_acc += x['val_acc']
        epoch_loss = epoch_loss/val_len
        epoch_acc = epoch_acc/val_len
        return {'val_loss': epoch_loss, 'val_acc': epoch_acc}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))

def append_value(dict_obj, key, value):
    if key in dict_obj:
        if not isinstance(dict_obj[key], list):
            dict_obj[key] = [dict_obj[key]]
        dict_obj[key].append(value)
    else:
        dict_obj[key] = value

def accuracy(outputs, labels):
    _, preds = T.max(outputs, dim=1)
    return T.tensor(T.sum(preds == labels).item())

  
def evaluate(model, val_loader,val_len):
    model.eval()
    outputs = [model.validation_step(batch) for k,batch in enumerate(val_loader)]
    return model.validation_epoch_end(outputs,val_len)

  
def fit(epochs, model, train_loader, val_loader,train_len,val_len, scheduler, opt_func):
    
    result = {}
    optimizer = opt_func
    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0
        running_train_acc = 0
        for l,batch in enumerate(train_loader):
            if l%20 == 0:
                print(l)
            loss, acc = model.training_step(batch)
            running_train_acc += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += (loss*len(batch[0])).detach().item()

            
        val = evaluate(model, val_loader,val_len)
        val['train_loss'] = (running_loss/train_len)
        val['train_acc'] = (running_train_acc/train_len)
        append_value(result, 'train_loss',(running_loss/train_len))
        append_value(result, 'train_acc', (running_train_acc/train_len))
        append_value(result, 'val_loss', val['val_loss'])
        append_value(result, 'val_acc', val['val_acc'])
        append_value(result, 'lr', optimizer.param_groups[0]['lr'])
        model.epoch_end(epoch, val)

        
        scheduler.step(val['val_loss'])
    return result



class EAMBlock(nn.Module):
    def __init__(self, kernel_size_s,kernel_size_k,in_channels=1,out_channels=1,k=3):
        super(EAMBlock,self).__init__()
        self.kernel_size_s = kernel_size_s
        self.gap = nn.AvgPool3d(self.kernel_size_s)
        self.gmp = nn.MaxPool3d(self.kernel_size_s)
        self.conv1d = nn.Conv1d(in_channels, out_channels, k,padding='same')
        self.sigmoid = nn.Sigmoid()
        
        self.kernel_size_k = kernel_size_k
        self.gap_k = nn.AvgPool3d(self.kernel_size_k)
        self.gmp_k = nn.MaxPool3d(self.kernel_size_k)
        
    def forward(self, x):    
        tmp_s = x
        x1 = self.gap(x)
        x2 = self.gmp(x)
        op_s = T.add(x1,x2)
        op_s = T.squeeze(op_s,2)
        op_s = T.squeeze(op_s,3)
        op_s = T.permute(op_s,(0,2,1))
        op_s = self.conv1d(op_s)
        op_s = self.sigmoid(op_s)
        op_s = T.squeeze(op_s,1)
        op_s = op_s[:,:,None,None,None]
        op_s = T.mul(tmp_s,op_s)
#         print(op_s.shape)
        
        tmp_t = T.permute(op_s,(0,2,1,3,4))
        
#         print(tmp_t.shape)
        x3 = self.gap_k(tmp_t)
        x4 = self.gmp_k(tmp_t)
        op_t = T.add(x3,x4)
        op_t = T.squeeze(op_t,2)
        op_t = T.squeeze(op_t,3)
        op_t = T.permute(op_t,(0,2,1))
        op_t = self.conv1d(op_t)
        op_t = self.sigmoid(op_t)
        op_t = T.squeeze(op_t,1)
        op_t = op_t[:,:,None,None,None]
        op_t = T.mul(tmp_t,op_t)
        return T.permute(op_t,(0,2,1,3,4))

class Attention_resnet18(DiseaseClassifier):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.resnet18 = models.video.r3d_18(pretrained=True)
        # Replace last layer
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 128)
        self.layer_count = -1
        for names,module in self.resnet18._modules.items():
            self.layer_count+=1
            self.add_module(f'layer{self.layer_count}',module)
            
        self.softmax = nn.Softmax(1)
        self.final = nn.Linear(128,num_classes)
        self.relu = nn.ReLU(inplace=True)
        del self.resnet18
        
        
        self.eam1 = EAMBlock((16,32,32),(64,32,32))
        self.eam2 = EAMBlock((8,16,16),(128,16,16))
        self.eam3 = EAMBlock((4,8,8),(256,8,8))
        self.eam4 = EAMBlock((2,4,4),(512,4,4))

        
    def forward(self, x):
        x = x.reshape(x.shape[0],x.shape[4],x.shape[1], x.shape[2], x.shape[3])
        x = self._modules['layer0'](x)
        x = self._modules['layer1'](x)
        x = self.eam1(x)
        x = self._modules['layer2'](x)
        x = self.eam2(x)
        x = self._modules['layer3'](x)
#         x = self.eam3(x)
        x = self._modules['layer4'](x)
#         x = self.eam4(x)    
        x = self._modules['layer5'](x)
        x = T.squeeze(x,4)
        x = T.squeeze(x,3)
        x = T.squeeze(x,2)
        x = self._modules['layer'+str(self.layer_count)](x)
        x = self.final(x)
        x = self.softmax(x)
        return x
    


model = Attention_resnet18()

for name,param in model.named_parameters():
    if 'layer4' in name or 'layer6' in name or 'eam' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
        

model = model.to(device)

num_epochs = 20
lr = 0.001
opt_func = T.optim.SGD(model.parameters(),lr,weight_decay=0.0001)
scheduler = ReduceLROnPlateau( opt_func,factor=0.1,  patience=2,  min_lr=lr * 0.01)

histories = []
test_accs = []
for i in range(1):
    print('Fold : ',i+1)
    train_classification_dataloader, val_classification_dataloader,test_classification_dataloader, X_test, y_test , X_train_lbl, X_val_lbl, X_test_lbl, y_test_lbl = rolled_DataLoaders(als, hc, ps)
    history = fit(num_epochs, model, train_classification_dataloader, val_classification_dataloader,len(X_train_lbl),len(X_val_lbl), scheduler, opt_func)
#     print(histories[i])
    acc = 0
    gts = []
    preds = []
    for j,batch in enumerate(test_classification_dataloader): 
        images, labels,features = batch
        images = images.to(device)
        labels = labels.to(device)
        out = model((images/255.))   
        acc += accuracy(out, labels)
        _,pred = T.max(out, dim=1)
        gts.append(labels.cpu().item())
        preds.append(pred.cpu().item())
    print('Ground truth:',gts)
    print('Prediction:',preds)
    print('Test accuracy:',(acc/len(test_classification_dataloader)).item())



