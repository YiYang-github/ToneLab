"""
model.py
machine learning models
"""

#[N, 1, 224, 224]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG(nn.Module):
    def __init__(self, pretrained=True, normalize_values=True, classes=3):
        super(VGG, self).__init__()
        self.vgg = models.vgg16(pretrained=pretrained)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.normalize_values = normalize_values
        
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Identity()
        self.classifier_values = nn.Linear(num_ftrs, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg(x)
        values = self.classifier_values(x)
        if self.normalize_values:
            values = self.sigmoid(values)
        return values

    def get_features_by_index(self, x, layer_index):
        """Extract the features from a specified layer index."""
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.vgg.features[layer_index].register_forward_hook(get_activation(str(layer_index)))
        _ = self.forward(x)
        return activation[str(layer_index)]

class ResNet(nn.Module):
    def __init__(self, pretrained=True, normalize_values=True, classes=3):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier_values = nn.Linear(num_ftrs, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        values = self.classifier_values(x)
        if self.normalize_values:
            values = self.sigmoid(values)
        return values

    def get_features_by_index(self, x, layer_index):
        """Extract the features from a specified layer index."""
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        layer = list(self.resnet.children())[layer_index]
        layer.register_forward_hook(get_activation('feature'))
        _ = self.forward(x)
        return activation['feature']

class DenseNet(nn.Module):
    def __init__(self, pretrained=True, normalize_values=True, classes=3):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.classifier_values = nn.Linear(num_ftrs, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.densenet.features(x)
        out = F.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        values = self.classifier_values(out)
        if self.normalize_values:
            values = self.sigmoid(values)
        return values

    def get_features_by_index(self, x, layer_index):
        """Extract the features from a specified layer index."""
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        layer = list(self.densenet.features.children())[layer_index]
        layer.register_forward_hook(get_activation('feature'))
        _ = self.forward(x)
        return activation['feature']


class mlp(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=3, normalize_values=True):
        super(mlp, self).__init__()
        self.normalize_values = normalize_values

        # Creating a list of layers
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU(inplace=True))  # Using ReLU activation function
            last_size = size
        layers.append(nn.Linear(last_size, output_size))  # Output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.normalize_values:
            # Normalize values to [0, 1]
            x = torch.sigmoid(x)
        return x