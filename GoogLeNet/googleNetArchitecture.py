import torch
import torch.nn as nn
from helper_layers import InceptionLayer, aux_classifier

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=100, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.lrn2 = nn.LocalResponseNorm(5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = InceptionLayer(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionLayer(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = InceptionLayer(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionLayer(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionLayer(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionLayer(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionLayer(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = InceptionLayer(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionLayer(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if aux_logits:
            self.aux1 = aux_classifier(512, num_classes)
            self.aux2 = aux_classifier(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.lrn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.training and self.aux_logits else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.training and self.aux_logits else None
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x
