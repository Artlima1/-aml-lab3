import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False):
        super(ResBlock, self).__init__()
        stride=(2 if downsample else 1)
        self.F = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        return torch.relu(self.F(x) + self.shortcut(x))
        
        


class TinyImgNetResNet(nn.Module):
    def __init__(self):
        super(TinyImgNetResNet, self).__init__()
        
        self.ResNet = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
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
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classification_head = nn.Linear(512, 200)
        
        
    def forward(self, x):
        x = self.ResNet(x)
        logits = self.classification_head(x)
        return logits