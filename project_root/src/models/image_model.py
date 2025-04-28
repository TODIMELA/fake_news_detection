import torch
import torch.nn as nn
import torchvision.models as models
 
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        """
        Initializes the ResNet classifier model.
        Args:
            num_classes (int): The number of output classes (default: 2 for binary classification).
        """
        super(ResNetClassifier, self).__init__()
        # Load the pretrained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        # Get the number of features in the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        # Replace the final fully connected layer for classification
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Forward pass of the ResNet classifier.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.resnet(x)

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=2):
        """
        Initializes the VGG classifier model.
        Args:
            num_classes (int): The number of output classes (default: 2 for binary classification).
        """
        super(VGGClassifier, self).__init__()
        # Load the pretrained VGG16 model
        self.vgg = models.vgg16(pretrained=True)
        # Get the number of features in the final fully connected layer
        num_ftrs = self.vgg.classifier[6].in_features
        # Replace the final fully connected layer for classification
        self.vgg.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Forward pass of the VGG classifier.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.vgg(x)