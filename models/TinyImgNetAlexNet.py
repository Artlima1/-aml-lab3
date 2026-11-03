import torch
from torch import nn


class TinyImgNetAlexNet(nn.Module):
    def __init__(self):
        super(TinyImgNetAlexNet, self).__init__()
        
        self.AlexNet = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2),     # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),             # pool1
            nn.Conv2d(64, 192, 5, 1, 2),    # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),             # pool2
            nn.Conv2d(192, 384, 3, 1, 1),   # conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),   # conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),   # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),             # pool5
            nn.Flatten()
        )
        self.classification_head = nn.Linear(9216, 200)
        
         
    def forward(self, x):
        x = self.AlexNet(x)
        logits = self.classification_head(x)
        return logits