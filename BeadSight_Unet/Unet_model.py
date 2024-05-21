
import torch
import torch.nn as nn
from torch.nn import functional as F

from PIL import Image
from torchvision import transforms
import os

from torchsummary import summary


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.InstanceNorm2d(C_out),
            nn.ELU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.InstanceNorm2d(C_out),
            nn.ELU(),
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

    def __init__(self):
        super(UNet, self).__init__()

        # 4 times
        self.C1 = Conv(45, 16)    #256    # Modified to take 15 channel inputs
        self.D1 = DownSampling(16) #128
        self.C2 = Conv(16, 32) # 128
        self.D2 = DownSampling(32) # 64
        self.C3 = Conv(32, 64) # 64
        self.D3 = DownSampling(64) # 32
        self.C4 = Conv(64, 128) # 32

        self.U2 = UpSampling(128)
        self.C7 = Conv(128, 64)
        self.U3 = UpSampling(64)
        self.C8 = Conv(64, 32)
        self.U4 = UpSampling(32)
        self.C9 = Conv(32, 16)


        self.Th = torch.nn.ReLU()  # Remove sigmoid activation

        # Separate convolutional layers for prediction
        self.pred = torch.nn.Conv2d(16, 1, 3, 1, 1)  



    def forward(self, x):
        # x: 45 x 256 x 256
        R1 = self.C1(x) # 16 x 256 x 256
        R2 = self.C2(self.D1(R1)) # 32 x 128 x 128
        R3 = self.C3(self.D2(R2)) # 64 x 64 x 64
        Y1 = self.C4(self.D3(R3)) #


        O2 = self.C7(self.U2(Y1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        
        return self.Th(self.pred(O4))  




if __name__ == '__main__':
    a = torch.randn(2, 45, 256, 256)
    net = UNet()
    print(net(a).shape)  # torch.Size([2, 1, 256, 256])

    # Generate the UNet structure figure
    summary(net, (45, 256, 256), device='cpu')
