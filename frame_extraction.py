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
from statistics import mean, median, mode
from torch.autograd import Variable
from sklearn.decomposition import PCA
from torchvision.utils import make_grid
!pip install openpyxl
import pandas as pd

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

# %% [markdown]
# **FAN**

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

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(device)

model = FAN(4)

model.load_state_dict(T.load('../input/model-losses/epoch_50_test_loss_0.06192380977452545.pt')['state_dict'])
def get_preds_fromhm(hm):
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = np.argmax(hm_reshape, axis=-1)
    scores = np.take_along_axis(hm_reshape, np.expand_dims(idx, axis=-1), axis=-1).squeeze(-1)
    preds = _get_preds_fromhm(hm, idx)

    return preds, scores


def _get_preds_fromhm(hm, idx):

    B, C, H, W = hm.shape
    idx += 1
    preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1

    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j] += np.sign(diff) * 0.25

    preds -= 0.5

    return preds

als_weights = pd.read_excel('../input/d/harneet02/dataset-weights/Dataset_weights.xlsx', sheet_name='ALS') #dataframe with details of class videos
hc_weights = pd.read_excel('../input/d/harneet02/dataset-weights/Dataset_weights.xlsx', sheet_name='HC')
ps_weights = pd.read_excel('../input/d/harneet02/dataset-weights/Dataset_weights.xlsx', sheet_name='PS')

path = '../input/als-videos/'
video_names = []
for j, video_name in enumerate(sorted(os.listdir(path))):
    als_weights.loc[j, 'Name'] = video_name
    
path = '../input/stroke-videos/'
video_names = []
for j, video_name in enumerate(sorted(os.listdir(path))):
    ps_weights.loc[j, 'Name'] = video_name
    
path = '../input/healthy-videos/'
video_names = []
for j, video_name in enumerate(sorted(os.listdir(path))):
    hc_weights.loc[j, 'Name'] = video_name

def get_frames_from_video(path, info):
    
    cap = cv2.VideoCapture(path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(totalframecount)
    #print(fps)
    skip_start_sec = int(info['Start clip'].item())
    skip_end_sec = int(info['End clip'].item())
    skip_end_frames = fps*skip_end_sec
    num_frames_one_task = (float(info['length of task'].item())/float(info['task in 1 length'].item()))*fps
    if skip_start_sec !=0:
        cap.set(1, skip_start_sec*fps)   #1 is cv2.CAP_PROP_POS_FRAMES

    count = 0 + round(skip_start_sec*fps)
    frames = []
            
    while cap.isOpened():
        if len(frames) >= totalframecount - skip_end_frames:
            break
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            count = count + 1 
            del frame
            cap.set(1, count)
            
        else:
            break

    cap.release()
    
    return frames,fps,num_frames_one_task

def equalize_frame_number(one_vid,fps,num_frames_one_task):
    length = len(one_vid)
    one_vid= np.array(one_vid)
    all_vids = []
    num_frames_one_task = int(round(num_frames_one_task))
    start = 0
    while(length>=num_frames_one_task+10):
        tmp_vid = one_vid[start : (start+num_frames_one_task+10)]
        length = length - num_frames_one_task-10
        start = start+num_frames_one_task+10
        all_vids.append(tmp_vid)
    return all_vids

def get_rect_frames(model, device, name , p1, p2, p3, p4 ):
    
    model.to(device)
    cr_vids = []
    lbls = []
    
    if name == 'ALS':
        path = '../input/als-videos/'
        df = als_weights.copy()
    elif name == 'PS':
        path = '../input/stroke-videos/'
        df = ps_weights.copy()
    elif name == 'HC':
        path = '../input/healthy-videos/'
        df = hc_weights.copy()
        
    for j, video_name in enumerate(sorted(os.listdir(path))):
        if df.loc[df['Name'] == video_name]['task in 1 length'].item() != 'Corrupt':
            
            video_frames,fps,num_frames_one_task = get_frames_from_video(path+video_name, df.loc[df['Name'] == video_name])
            print(len(video_frames),fps,num_frames_one_task)
            one_vid= []
            for i, (img) in tqdm(enumerate(video_frames)):
                img_temp = img
                img = cv2.resize(img, (256, 256))
                img = T.from_numpy(img)
                img = T.div(img, 255.0).type(T.FloatTensor).to(device)
                img = img.permute(2, 0, 1)
                img = img.unsqueeze(0)
                label_hat = model(img)
                hmaps = np.expand_dims(label_hat[3][0].to('cpu').detach().numpy(),axis=0)
                preds,_ = get_preds_fromhm(hmaps)
           
                img_temp = cv2.resize(img_temp, (64, 64))
                
                pts1 = np.float32([[preds[0][p1][0],preds[0][p2][1]],[preds[0][p4][0],preds[0][p2][1]],[preds[0][p1][0],preds[0][p3][1]],[preds[0][p4][0],preds[0][p3][1]]])
                pts2 = np.float32([[0,0],[64,0],[0,64],[64,64]])

                matrix = cv2.getPerspectiveTransform(pts1,pts2)
                result = cv2.warpPerspective(img_temp, matrix, (64,64))
#                 print(np.min(result),np.max(result))
                
                one_vid.append(np.array(result))
                

                del img_temp, img, hmaps, preds, result, matrix, label_hat

                #### for visualization
                
#                 fig, ax = plt.subplots(1)
#                 ax.imshow(result)
#                 ax.axis('off')
#                 circ = Circle( (preds[0][p1][0],preds[0][p1][1]), 0.3, color='white')
#                 ax.add_patch(circ)
#                 circ = Circle( (preds[0][p2][0],preds[0][p2][1]), 0.3, color='white')
#                 ax.add_patch(circ)
#                 circ = Circle( (preds[0][p3][0],preds[0][p3][1]), 0.3, color='white')
#                 ax.add_patch(circ)
#                 circ = Circle( (preds[0][p4][0],preds[0][p4][1]), 0.3, color='white')
#                 ax.add_patch(circ)
#                 plt.show()
#                 break
                

            all_vids = equalize_frame_number(one_vid,fps,max(16,num_frames_one_task))
            all_vids = np.array(all_vids)
            new_all_vids = []
#             print(all_vids.shape)
            for m in range(all_vids.shape[0]):
                extract = 0
                step = math.floor(all_vids.shape[1]/16)
                tmp_new_vid = []
                while (len(tmp_new_vid)!=16):
                    tmp_new_vid.append(all_vids[m][extract])
                    extract = extract+step
                new_all_vids.append(np.array(tmp_new_vid))
                lbls.append(video_name)
            new_all_vids = np.array(new_all_vids)
            cr_vids.extend(new_all_vids)

            
            print(j)
            print(np.array(cr_vids).shape)
            if j==75:
                break
            
            del one_vid, video_frames

    return cr_vids, lbls

def show_batch(frames):
    images = T.from_numpy(np.array(frames)).permute(0, 3, 1,2)
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=12).permute(1, 2, 0))
    plt.show()


X_ALS_patches ,lbls_ALS = get_rect_frames(model, device, 'ALS' , 0, 19,8, 16 )

np.save('X_ALS_pch', np.array(X_ALS_patches))
np.save('ALS_lbls', np.array(lbls_ALS))

X_PS_patches, lbls_PS = get_rect_frames(model, device, 'PS' , 0, 19,8, 16 )

np.save('X_PS_pch', np.array(X_PS_patches))
np.save('PS_lbls', np.array(lbls_PS))

X_HC_patches, lbls_HC = get_rect_frames(model, device, 'HC' ,0, 19,8, 16)

np.save('X_HC_pch', np.array(X_HC_patches))
np.save('HC_lbls', np.array(lbls_HC))

show_batch(X_HC_patches[1]/255.)

