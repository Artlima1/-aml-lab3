import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False):
        super(ResBlock, self).__init__()
        self.F = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=(2 if downsample else 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(),
        )
        
    def forward(self, x):
        logits = nn.ReLU(self.F(x) + x)
        return logits
        
        


class TinyImgNetResNet(nn.Module):
    def __init__(self):
        super(TinyImgNetResNet, self).__init__()
        
        self.ResNet = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.MaxPool2d(3, 1),
            
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            
            ResBlock(64, 128, True),
            ResBlock(128, 128, False),
            ResBlock(128, 128, False),
            
            ResBlock(128, 256, True),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            
            ResBlock(256, 512, True),
            ResBlock(512, 512, False),
            ResBlock(512, 512, False),
            
            nn.MaxPool2d(3, 1),
            nn.Flatten(),
            nn.Linear(7168, 1000)
        )
        
        
    def forward(self, x):
        logits = self.ResNet(x)
        return logits