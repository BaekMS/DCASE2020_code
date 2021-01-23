import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import random
from sklearn.preprocessing import minmax_scale
from time import time
import pandas as pd
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import shutil
from multiprocessing import Process
import math
import torchvision.models
import adabound
import custom_transformer

class ResBlock(nn.Module):
    def __init__(self, i):
        super(ResBlock, self).__init__()
        # if i==0 or i==1:
        #     shape=64
        #     # self.main_model = custom_transformer.Transformer(d_model=shape, nhead=int(shape/8), dim_feedforward=1024,
        #     #                                                  num_encoder_layers=2, num_decoder_layers=6)
        # elif i==2:
        #     shape=128
        #
        # self.main_model = custom_transformer.Transformer(d_model=shape, nhead=8, dim_feedforward=512,
        #                                                  num_encoder_layers=4, num_decoder_layers=6)
        self.avp1 = nn.AvgPool2d((1, 4), stride=(1, 4), padding=(0, 1))
        self.bn1=nn.BatchNorm2d(3)
        self.cv1=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=(1,2), padding=(2,1), dilation=(2,1))
        self.bn2=nn.BatchNorm2d(32)
        self.cv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=(1,2), padding=(2,1), dilation=(2,1))


        self.avp2 = nn.AvgPool2d((1, 4), stride=(1, 4), padding=(0, 1))

        self.bn3 = nn.BatchNorm2d(64)
        self.cv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 2), padding=(2, 1),
                             dilation=(2, 1))
        self.bn4=nn.BatchNorm2d(128)
        self.cv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=(1, 2), padding=(2, 1),
                             dilation=(2, 1))


    def forward(self, x):
        # x = x.squeeze(1).permute(2, 0, 1)
        #print(x.shape)
        # pd=self.main_model(x,x).permute(1,2,0).unsqueeze(1)
        #
        # x=x.permute(1,2,0).unsqueeze(1)
        # # x=x.permute(1,2,0).unsqueeze(1)+pd
        # x=torch.cat((x,pd), dim=1)


        avg1=self.avp1(x).to('cuda')

        pad1=(0,0, 0,0, 0,61, 0,0)

        avg1=F.pad(avg1, pad1, 'constant', 0)


        x=self.bn1(x)
        x=F.relu(x)
        x=self.cv1(x)




        # x = (x - x.mean(dim=(1,2,3),keepdim=True))/x.std(dim=(1,2,3),keepdim=True)
        x=self.bn2(x)
        x=F.relu(x)
        x=self.cv2(x)




        x+=avg1


        avg2=self.avp2(x).to('cuda')

        pad2 = (0, 0, 0, 0, 0,192, 0, 0)
        avg2 = F.pad(avg2, pad2, 'constant', 0)


        x = self.bn3(x)
        x = F.relu(x)
        x = self.cv3(x)
        x=self.bn4(x)



        x = F.relu(x)
        x = self.cv4(x)


        x+=avg2

        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.resmodule = nn.ModuleList([ResBlock(i) for i in range(3)])
        self.bn1=nn.BatchNorm2d(256)
        self.cv1=nn.Conv2d(in_channels=256, out_channels=768, kernel_size=1)

        self.bn2=nn.BatchNorm2d(768)
        self.cv2=nn.Conv2d(in_channels=768, out_channels=10, kernel_size=1)
        self.bn3=nn.BatchNorm2d(10)

        self.avp = nn.AvgPool2d((256, 27))


    def forward(self, x):

        x=torch.cat((self.resmodule[0](x[:,:,0:64,:]), self.resmodule[1](x[:,:,64:128,:]), self.resmodule[2](x[:,:,128:256,:])), dim=2)


        x=self.bn1(x)

        x=F.relu(x)

        x=self.cv1(x)


        x=self.bn2(x)
        x=self.cv2(x)

        x = self.avp(x)
        x = x.view(-1, 10)

        x=F.log_softmax(x, dim=-1)

        return x