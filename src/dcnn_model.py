import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_config

class GraspDCNN(nn.Module):
    def __init__(self, config_path="config/config.yaml"):
        super(GraspDCNN, self).__init__()
        
        config = load_config(config_path)
        net_config = config['network']
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, net_config['conv1_filters'], kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(size=9, alpha=0.001/9.0, beta=0.75, k=1.0)
        
        self.conv2 = nn.Conv2d(net_config['conv1_filters'], net_config['conv2_filters'], kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.LocalResponseNorm(size=9, alpha=0.001/9.0, beta=0.75, k=1.0)
        
        # Calculate flattened size: 200 -> 100 -> 50
        self.flattened_size = net_config['conv2_filters'] * 50 * 50
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, net_config['fc1_size'])
        self.fc2 = nn.Linear(net_config['fc1_size'], net_config['fc2_size'])
        self.output = nn.Linear(net_config['fc2_size'], net_config['output_size'])
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.output(x)
        
        return x

class GraspLoss(nn.Module):
    def __init__(self, config_path="config/config.yaml"):
        super(GraspLoss, self).__init__()
        
        config = load_config(config_path)
        loss_config = config['loss']
        
        # Updated normalization constants
        self.theta = loss_config['theta_loss']  # Should be ~0.2
        self.omega = loss_config['omega_loss']  # Should be ~1.0
        
    def forward(self, predicted, target):
        batch_size = predicted.size(0)
        
        # Split position and rotation
        pred_pos = predicted[:, :3]
        pred_rot = predicted[:, 3:]
        
        target_pos = target[:, :3]
        target_rot = target[:, 3:]
        
        # Compute losses with proper scaling
        pos_diff = torch.norm(pred_pos - target_pos, dim=1)
        rot_diff = torch.norm(pred_rot - target_rot, dim=1)
        
        # Normalized losses (should be in range [0, 10])
        pos_loss = torch.mean((pos_diff / self.theta) ** 2)
        rot_loss = torch.mean((rot_diff / self.omega) ** 2)
        
        total_loss = pos_loss + rot_loss
        
        return total_loss, pos_loss, rot_loss
