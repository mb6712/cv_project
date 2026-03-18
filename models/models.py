from bcos_layer import BcosConv2d,Bcoslin
from torch import nn
import torch

class CNNBcos(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.C=7
        # Convolution layers
        self.features = nn.Sequential(
            BcosConv2d(1, 6, kernel_size=(5,5), padding=2)
            ,nn.BatchNorm2d(6)
            ,nn.ReLU(inplace=True)
            ,nn.AvgPool2d(kernel_size=2,stride=2)
            
            ,BcosConv2d(6, 16, kernel_size=(10,10))
            ,nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            Bcoslin(784,16,),
            
            Bcoslin(16, self.C, ),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x
class CNN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.C=7
        # Convolution layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5,5), padding=2)
            ,nn.BatchNorm2d(6)
            ,nn.ReLU(inplace=True)
            ,nn.AvgPool2d(kernel_size=2,stride=2)
            
            ,nn.Conv2d(6, 16, kernel_size=(10,10))
            ,nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(784,16,),
            
            nn.Linear(16, self.C, ),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x
class Vgg(nn.Module):
    def __init__(self, drop=0.2):
        super().__init__()
        self.C=7
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(4608, 4096)
        self.lin2 = nn.Linear(4096, 4096)
        self.lin3 = nn.Linear(4096, self.C)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        #x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        #x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        #x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        x = F.relu(self.bn4a(self.conv4a(x)))
        #x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)
        #print(x.shape)

        #x = x.view(-1, 512 * 2 * 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.drop(self.lin1(x)))
        #x = F.relu(self.drop(self.lin2(x)))
        x = self.lin3(x)
        return x