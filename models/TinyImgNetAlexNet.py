import torch
from torch import nn


class TinyImgNetAlexNet(nn.Module):
    def __init__(self):
        super(TinyImgNetAlexNet, self).__init__()
        
        self.AlexNet = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2),     # conv1
            nn.MaxPool2d(3, 2),             # pool1
            nn.Conv2d(64, 192, 5, 1, 2),    # conv2
            nn.MaxPool2d(3, 2),             # pool2
            nn.Conv2d(192, 384, 3, 1, 1),   # conv3
            nn.Conv2d(384, 256, 3, 1, 1),   # conv4
            nn.Conv2d(256, 256, 3, 1, 1),   # conv5
            nn.MaxPool2d(3, 2),             # pool5
            nn.Flatten()
        )
         
    def forward(self, x):
        logits = self.AlexNet(x)
        return logits