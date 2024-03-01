import torch
import torch.nn as nn
import torchvision.models as models
import argparse
'''
class Detached_ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, input_ch=3):
        super().__init__()

        # Load the pretrained ResNet model
        resnet_model = models.resnet18(pretrained=pretrained)
        resnet_model.conv1 = nn.Conv2d(input_ch, resnet_model.conv1.weight.shape[0], 3, 1, 1, bias=False) 
        resnet_model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        # Isolate the feature extraction layers
        self.features = nn.Sequential(*list(resnet_model.children())[:-1])

        class ClassifierLayer(nn.Module):
            def __init__(self, in_features=resnet_model.fc.in_features, out_features=num_classes):
                super().__init__()
                self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
                self.bias = nn.Parameter(torch.Tensor(out_features))
                nn.init.kaiming_uniform_(self.weight, a=5)
                nn.init.constant_(self.bias, 0)

            def forward(self, x):
                x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
                return torch.nn.functional.linear(x_norm, self.weight, self.bias)
        
        self.classifier = ClassifierLayer(resnet_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
'''
class Detached_ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10,input_ch=3):
        super(Detached_ResNet, self).__init__()

        # Load the pretrained ResNet model
        resnet_model = models.resnet50(pretrained=pretrained)
        resnet_model.conv1 = nn.Conv2d(input_ch, resnet_model.conv1.weight.shape[0], 3, 1, 1, bias=False) 
        resnet_model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        # Isolate the feature extraction layers
        self.features = nn.Sequential(*list(resnet_model.children())[:-1])

        # Isolate the classifier layer
        self.classifier = nn.Linear(resnet_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

