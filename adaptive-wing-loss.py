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
from torch.optim.lr_scheduler import MultiStepLR,StepLR
from scipy.ndimage.morphology import grey_dilation

img_paths = ['../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/ALS/Frames/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Healthy/Frames/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Post_Stroke/Frames/']
landmark_paths = ['../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/ALS/Landmarks_gt/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Healthy/Landmarks_gt/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Post_Stroke/Landmarks_gt/']
X = []
Y = []
for l in range(len(img_paths)):
    img_path = img_paths[l]
    landmark_path = landmark_paths[l]
    for landmark_file in os.listdir(landmark_path):
        if landmark_file.startswith(('A002_02','A006_02','A008_02','N001_02','N002_02','N003_02','OP01_02','OP02_02', 'OP03_02')):
            continue
        with open(landmark_path+landmark_file) as f:
            lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            frame_name = landmark_file.split('.')[0]+'.avi_'+ str(line.split(',')[0]) +'.jpg'
            X.append(cv2.imread(img_path+frame_name))
            label = []
            for i in range(1,137,2):
                label.append((float(line.split(',')[i]),float(line.split(',')[i+1])))
            Y.append(label)

img_paths = ['../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/ALS/Frames/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Healthy/Frames/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Post_Stroke/Frames/']
landmark_paths = ['../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/ALS/Landmarks_gt/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Healthy/Landmarks_gt/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Post_Stroke/Landmarks_gt/']
X_val = []
Y_val = []
for l in range(len(img_paths)):
    img_path = img_paths[l]
    landmark_path = landmark_paths[l]
    for landmark_file in os.listdir(landmark_path):
        if not landmark_file.startswith(('A002_02','A006_02','N001_02','N002_02','OP01_02','OP02_02')):
            continue
        with open(landmark_path+landmark_file) as f:
            lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            frame_name = landmark_file.split('.')[0]+'.avi_'+ str(line.split(',')[0]) +'.jpg'
            X_val.append(cv2.imread(img_path+frame_name))
            label = []
            for i in range(1,137,2):
                label.append((float(line.split(',')[i]),float(line.split(',')[i+1])))
            Y_val.append(label)

img_paths = ['../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/ALS/Frames/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Healthy/Frames/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Post_Stroke/Frames/']
landmark_paths = ['../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/ALS/Landmarks_gt/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Healthy/Landmarks_gt/','../input/neuronet/NeuroNet-20210916T201432Z-001/NeuroNet/Post_Stroke/Landmarks_gt/']
X_test = []
Y_test = []
for l in range(len(img_paths)):
    img_path = img_paths[l]
    landmark_path = landmark_paths[l]
    for landmark_file in os.listdir(landmark_path):
        if not landmark_file.startswith(('A008_02','N003_02','OP03_02')):
            continue
        with open(landmark_path+landmark_file) as f:
            lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            frame_name = landmark_file.split('.')[0]+'.avi_'+ str(line.split(',')[0]) +'.jpg'
            X_test.append(cv2.imread(img_path+frame_name))
            label = []
            for i in range(1,137,2):
                label.append((float(line.split(',')[i]),float(line.split(',')[i+1])))
            Y_test.append(label)

print(len(X),len(Y))
print(len(X_val),len(Y_val))
print(len(X_test),len(Y_test))

def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    #print('point:',point)
    ul = [math.floor(round(point[0]) - 3 * sigma), math.floor(round(point[1]) - 3 * sigma)]
    #print('ul:',ul)
    br = [math.floor(round(point[0]) + 3 * sigma), math.floor(round(point[1]) + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    #print('br:',br)
    #print('size:',size)
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    #print('img_x:',img_x)
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    #print('img_y:',img_y)
    #print('diff:',img_y[1]-(img_y[0]-1))
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image

class ALSDataset(Dataset):

    def __init__(self,X,Y,transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self,index):
        x = self.X[index]
        y = self.Y[index]
        if self.transform is not None:
            transformed = self.transform(image=x, keypoints=y)
            x = transformed['image']
            y = transformed['keypoints']

        y_pts = []
        y_ = 256
        x_ = 256
        target_size=64
        x_scale = target_size/x_
        y_scale = target_size/y_
        for j in range(68):
            y_pts.append((y[j][0]*x_scale,y[j][1]*y_scale))

        heatmaps = np.zeros((68, 64, 64), dtype=np.float32)
        M = np.zeros((68, 64, 64), dtype=np.float32)

        for k in range(68):
            heatmaps[k] = draw_gaussian(heatmaps[k],y_pts[k],2)
            dilate = grey_dilation(heatmaps[k] ,size=(3,3))
            weight_map = M[k]
            weight_map[np.where(dilate>0.2)] = 1
            M[k] = weight_map

        return x,heatmaps,M

class ConvBlock(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(ConvBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,int(out_planes/2),3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_planes/2))
        self.conv2 = nn.Conv2d(int(out_planes/2),int(out_planes/4),3,stride=1,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_planes/4))
        self.conv3 = nn.Conv2d(int(out_planes/4),int(out_planes/4),3,stride=1,padding=1,bias=False)


        if in_planes!=out_planes:
            self.downsample = nn.Sequential(
              nn.BatchNorm2d(in_planes),
              nn.ReLU(True),
              nn.Conv2d(in_planes,out_planes,1,stride=1,bias=False)
          )
        else:
            self.downsample=None

    def forward(self,x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1,True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2,True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3,True)
        out3 = self.conv3(out3)

        out3 = T.cat((out1,out2,out3),1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 = out3+residual
        return out3

class HourGlass(nn.Module):
    def __init__(self,num_modules,depth, num_features):
        super(HourGlass,self).__init__()
        self.num_modules = num_modules
        self.features = num_features
        self.depth = depth

        self._generate_network(self.depth)


    def _generate_network(self,level):
        self.add_module('b1_' + str(level) , ConvBlock(self.features,self.features))
        self.add_module('b2_' + str(level) , ConvBlock(self.features,self.features))

        if level>1:
            self._generate_network(level-1)
        else:
            self.add_module('b2_plus_' + str(level) , ConvBlock(self.features,self.features))
        self.add_module('b3_' + str(level) , ConvBlock(self.features,self.features))

    def _forward(self,level,inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        low1 = F.avg_pool2d(inp,2,stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level>1:
            low2 = self._forward(level-1,low1)
        else:
            low2=low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3=low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3,scale_factor=2,mode='nearest')
        return up1+up2

    def forward(self,x):
        return self._forward(self.depth,x)

class FAN(nn.Module):
    def __init__(self,num_modules=1):

        super(FAN,self).__init__()
        self.num_modules = num_modules

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64,128)
        self.conv3 = ConvBlock(128,128)
        self.conv4 = ConvBlock(128,256)

        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module) , HourGlass(1,4,256))
            self.add_module('top_m_' + str(hg_module) , ConvBlock(256,256))
            self.add_module('conv_last' + str(hg_module) , nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0))
            self.add_module('bn_end' + str(hg_module) , nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module) , nn.Conv2d(256,68,kernel_size=1,stride=1,padding=0))

            if hg_module<self.num_modules-1:
                self.add_module('bl' + str(hg_module) , nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0))
                self.add_module('al' + str(hg_module) , nn.Conv2d(68,256,kernel_size=1,stride=1,padding=0))

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)),True)
        x = F.avg_pool2d(self.conv2(x),2,stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []

        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg


            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)) , True)

            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules-1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' +str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs

from sklearn.model_selection import train_test_split

train_transform = A.Compose([
    A.Resize(256, 256),
    A.augmentations.geometric.rotate.Rotate(limit=30,p=0.5),
    A.augmentations.transforms.ColorJitter(p=0.5),
    A.augmentations.transforms.GaussNoise(p=0.5),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy'))

train_dataset = ALSDataset(X,Y,transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size = 16,shuffle =True)


val_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy'))
val_dataset = ALSDataset(X_val,Y_val,transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size = 16,shuffle=True)

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(device)

model = FAN(4)

from torch.utils.model_zoo import load_url
fan_weights = load_url('https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar', map_location=lambda storage, loc: storage)

model.load_state_dict(fan_weights)

def freeze():
    for name, param in model.named_parameters():
        if 'b2_2' in name or 'b2_1' in name or 'b2_plus_1' in name or 'b3_1' in name or 'b3_2' in name or 'b1_1' in name:
            param.requires_grad = False

class AWing(nn.Module):

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha   = float(alpha)
        self.omega   = float(omega)
        self.epsilon = float(epsilon)
        self.theta   = float(theta)

    def forward(self, y_pred , y):
        lossMat = T.zeros_like(y_pred)
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y)))*(self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        C = self.theta*A - self.omega*T.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        case1_ind = T.abs(y-y_pred) < self.theta
        case2_ind = T.abs(y-y_pred) >= self.theta
        lossMat[case1_ind] = self.omega*T.log(1+T.abs((y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind]*T.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
        return lossMat

class Loss_weighted(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWing(alpha, omega, epsilon, theta)

    def forward(self, y_pred , y, M):
        M = M.float()
        Loss = self.Awing(y_pred,y)
        weighted = Loss * (self.W * M + 1.)
        return weighted.mean()

model.to(device)
freeze()
criterion = Loss_weighted()
optimizer = T.optim.RMSprop(model.parameters(), lr=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[45,90], gamma=0.1)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
start_epoch = 1
num_epochs = 121
loss_values_train = []
loss_values_val = []
for epoch in range(start_epoch,num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)
    train_loss = 0.0
    model.train()
    for i, (img,label,M) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()    
        img = T.div(img, 255.0).to(device)
        label_hat = model(img)
        loss = 0.0
        for p in range(4):
            loss += criterion(label_hat[p],label.to(device),M.to(device))
        loss = loss/4
        loss.backward()
        optimizer.step()
        train_loss = train_loss + (loss.data.item()*img.shape[0])
    train_loss = train_loss/len(train_dataset)
    loss_values_train.append(train_loss)
  
    val_loss = 0.0
    model.eval()
    for i, (img,label,M) in tqdm(enumerate(val_dataloader)):
        img = T.div(img, 255.0).to(device)
        label_hat = model(img)
        loss = 0.0
        for p in range(4):
            loss += criterion(label_hat[p],label.to(device),M.to(device))
        loss = loss/4
        val_loss = val_loss + (loss.data.item()*img.shape[0])
    val_loss = val_loss/len(val_dataset)
    loss_values_val.append(val_loss)
    if epoch%10==0:
         T.save({          
                             "epoch": epoch,
                             "state_dict": model.state_dict(),
                             "val_loss": val_loss,
                             "train_loss":train_loss,
                             "optimizer": optimizer.state_dict()
                         },
                         'epoch:{}_val_loss:{}.pt'.format(epoch,val_loss))
    scheduler.step()

with open("train_loss.txt", "w") as output1:
    output1.write(str(loss_values_train))
with open("val_loss.txt", "w") as output2:
    output2.write(str(loss_values_val))


fig, axs = plt.subplots(1,figsize=(12,12))

axs.plot(loss_values_train,label="train")
axs.plot(loss_values_val,label="val")


axs.legend(loc='upper right')


fig.suptitle('Losses')

plt.show()
