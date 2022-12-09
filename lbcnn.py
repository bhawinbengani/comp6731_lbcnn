import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLBC(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)

class LayerLBC(nn.Module):
    def __init__(self, numChannels, numWeights, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(numChannels)
        self.conv_lbp = ConvLBC(numChannels, numWeights, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(numWeights, numChannels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.batch_norm(x)
        x = F.relu(self.conv_lbp(x))
        x = self.conv_1x1(x)
        x.add_(residual)
        return x
        
class LBCNN(nn.Module):
    def __init__(self, nInputPlane=3, numChannels=384, numWeights=704, full=512, depth=10, 
                 sparsity=0.001, num_classes=100):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(nInputPlane, numChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(numChannels),
            nn.ReLU(inplace=True)
        )
        
        lbc_layers = [LayerLBC(numChannels, numWeights, sparsity) for i in range(depth)]
        self.LBCLayer_block = nn.Sequential(*lbc_layers)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=5)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(numChannels * 6 * 6, full)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(full, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.LBCLayer_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout(x))
        x = F.relu(x)
        x = self.fc2(self.dropout(x))
        return x