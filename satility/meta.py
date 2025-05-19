# models/meta.py
import torch.nn as nn
import torch

class MetaModel(nn.Module):
    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble                     # ya congelado
        self.head = nn.Sequential(
            nn.Conv2d(6, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 4, 5, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, 1),                       # sin Sigmoid
            nn.Sigmoid()                             # [B,1,H,W]
        )

    def forward(self, x):
        with torch.no_grad():                        # evita grad en ensamble
            feats = self.ensemble(x)                 # [B,6,H,W]
        return self.head(feats)                      # [B,1,H,W]
