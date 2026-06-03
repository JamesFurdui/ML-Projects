import torch
import torchvision
import torch.nn as nn   

class InceptionLayer(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, 
                 out_3x3, out_5x5_reduce, out_5x5, out_pool_proj):
        super().__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            
            nn.Conv2d(out_3x3_reduce, out_3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_3x3),
            
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_5x5_reduce),
            
            nn.Conv2d(out_5x5_reduce, out_5x5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_5x5),
            
        )

        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool_proj, kernel_size=1),
            nn.BatchNorm2d(out_pool_proj),
            
        )

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        p4 = self.path4(x)

        return torch.cat([p1, p2, p3, p4], dim=1)
    
class aux_classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x