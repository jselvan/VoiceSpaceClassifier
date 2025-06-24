import torch
import torch.nn as nn

class ConvSpeakerNet(nn.Module):
    def __init__(self, n_classes=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, n_classes)

    def extract_features(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.emb(x)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @classmethod
    def from_pretrained(cls, model_path, n_classes=16, device='cpu'):
        model = cls(n_classes=n_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model