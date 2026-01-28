"""
Neural network for chess position evaluation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleChessNet(nn.Module):
    """Neural network for chess evaluation"""
    def __init__(self):
        super().__init__()
        # Input: 13 channels (6 white pieces + 6 black pieces + turn)
        self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Value head (position evaluation)
        self.value_fc1 = nn.Linear(256 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x shape: (batch, 13, 8, 8)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Value head
        value = x.view(x.size(0), -1)  # Flatten
        value = torch.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = torch.relu(self.value_fc2(value))
        value = torch.tanh(self.value_out(value))  # Output between -1 and 1
        
        return value

class PositionEvaluator(nn.Module):
    """
    Even simpler model for just position evaluation
    Good for starting out
    """
    def __init__(self):
        super().__init__()
        # Input: 13 * 8 * 8 = 832 features
        self.fc1 = nn.Linear(13 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.output(x))
        return x

def test_model():
    """Test the model"""
    model = SimpleChessNet()
    
    # Create dummy input matching your board representation
    dummy_input = torch.randn(4, 13, 8, 8)  # Batch of 4 positions
    
    value, policy = model(dummy_input)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Value output shape: {value.shape}")
    print(f"Policy output shape: {policy.shape}")
    print(f"Sample evaluation: {value[0].item():.3f}")
    
    return model

if __name__ == "__main__":
    test_model()