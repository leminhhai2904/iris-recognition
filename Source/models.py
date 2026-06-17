import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DeepIrisResNet(nn.Module):
    def __init__(self, num_classes=224):
        super().__init__()
        # Load the pre-trained ResNet50 model trained on ImageNet[cite: 50, 63].
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Change the last layer to match the number of classes in the dataset[cite: 91].
        # The IIT Delhi dataset contains 224 different people[cite: 105].
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)