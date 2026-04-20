import torch
import torch.nn as nn
import torch.nn.functional as F

#torch.backends.cuda.matmul.allow_tf32 = True
#storch.backends.cuda.allow_tf32 = True


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, X):
        identity = self.skip(X)
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return F.relu(Y+identity)


class ResNet18_CIFAR(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l1 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
        )
        self.l2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128)
        )
        self.l3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256)
        )
        self.l4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, X):
        X = F.relu(self.bn1(self.Conv1(X)))

        X = self.pool1(X)
        
        X = self.l1(X)
        X = self.l2(X)
        X = self.l3(X)
        X = self.l4(X)

        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)
        return X

