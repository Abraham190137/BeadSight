import torch
import torch.nn as nn
from torch.nn import functional as F

from PIL import Image
from torchvision import transforms
import os

from torchsummary import summary

from typing import List, Tuple, Dict


class Conv(nn.Module):
    def __init__(self, C_in, C_out, dropout_prob):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 5, 1, 2),
            nn.InstanceNorm2d(C_out),
            nn.ELU(),
            nn.Dropout(dropout_prob),  # Add dropout

            nn.Conv2d(C_out, C_out, 5, 1, 2),
            nn.InstanceNorm2d(C_out),
            nn.ELU(),
            nn.Dropout(dropout_prob)  # Add dropout
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)


class UNet(nn.Module):

    def __init__(self, 
                 window_size: int,
                 dropout_prob: float = 0.25):  # Add dropout_prob parameter
        
        super(UNet, self).__init__()
        self.window_size = window_size # save the window size for later use
        
        # 4 times
        self.C1 = Conv(3*window_size, 16, dropout_prob)    #256    # Modified to take 15 channel inputs
        self.D1 = DownSampling(16) #128
        self.C2 = Conv(16, 32, dropout_prob) # 128
        self.D2 = DownSampling(32) # 64
        self.C3 = Conv(32, 64, dropout_prob) # 64
        self.D3 = DownSampling(64) # 32
        self.C4 = Conv(64, 128, dropout_prob) # 32

        self.U2 = UpSampling(128)
        self.C7 = Conv(128, 64, dropout_prob)
        self.U3 = UpSampling(64)
        self.C8 = Conv(64, 32, dropout_prob)
        self.U4 = UpSampling(32)
        self.C9 = Conv(32, 16, 0) # no dropout in the last layer

        self.act = torch.nn.ReLU() # use ReLU as activation function

        # Separate convolutional layers for prediction
        self.pred = torch.nn.Conv2d(16, 1, 3, 1, 1)  

    def forward(self, x):        
        # x: batch, 3, window_size, 256, 256
        # need to convert to batch, 3*window_size, 256, 256
        x = x.view(-1, 3*self.window_size, 256, 256)
        R1 = self.C1(x) # 16 x 256 x 256
        R2 = self.C2(self.D1(R1)) # 32 x 128 x 128
        R3 = self.C3(self.D2(R2)) # 64 x 64 x 64
        Y1 = self.C4(self.D3(R3)) #

        O2 = self.C7(self.U2(Y1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        
        return self.act(self.pred(O4)).squeeze(1)  

if __name__ == '__main__':
    a = torch.randn(2, 45, 256, 256)
    net = UNet(window_size=15, dropout_prob=0.3)
    print(net(a).shape)  # torch.Size([2, 1, 256, 256])

    # Generate the UNet structure figure
    summary(net, (45, 256, 256), device='cpu')
