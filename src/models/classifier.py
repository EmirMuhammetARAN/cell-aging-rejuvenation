import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self, output_size):
        super(Classifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)

    def forward(self, x):
        return self.model(x)
    