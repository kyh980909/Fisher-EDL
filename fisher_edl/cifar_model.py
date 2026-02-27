import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_p),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_p),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_cifar_model(backbone="simple", num_classes=10, dropout_p=0.0):
    if backbone == "simple":
        return SimpleCNN(num_classes=num_classes, dropout_p=dropout_p)
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        if dropout_p > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout_p)),
                nn.Linear(model.fc.in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown backbone: {backbone}")
