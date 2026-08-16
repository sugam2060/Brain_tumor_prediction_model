import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class BrainTumorVGG16(nn.Module):
    """
    PyTorch Brain Tumor Classification Model using VGG16 backbone.
    Inputs: Tensor of shape (batch_size, 3, 128, 128)
    Outputs: Logits of shape (batch_size, 4) for classes [glioma, meningioma, notumor, pituitary]
    """

    def __init__(self, num_classes: int = 4, freeze_features: bool = True, weights=None):
        super(BrainTumorVGG16, self).__init__()

        vgg = vgg16(weights=weights)

        self.features = vgg.features

        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
