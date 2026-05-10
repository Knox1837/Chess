"""Neural network for chess position evaluation"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Residual Block ─────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    Two conv layers with a skip connection.
    If the input and output channels differ, a 1x1 conv adapts the skip.
    Keeps gradients flowing through deep networks.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)   # skip connection
        return out


# ── Upgraded Model ─────────────────────────────────────────────────────────────
class ChessNet(nn.Module):
    """
    Residual CNN for chess position evaluation.
    Backwards compatible with SimpleChessNet:
      - Same input:  (batch, 13, 8, 8)
      - Same output: (batch, 1)  scalar in [-1, +1]
      - Same forward() signature

    Improvements over SimpleChessNet:
      - Residual blocks prevent vanishing gradients
      - Global average pooling replaces the 16k-neuron FC bottleneck
      - Deeper but cheaper: more capacity without more parameters
      - Lighter dropout (0.2) applied after pooling only
    """
    def __init__(self, num_filters=128, num_residual_blocks=6):
        super().__init__()

        # ── Stem: project 13 channels → num_filters ───────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(13, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # ── Residual tower ────────────────────────────────────────────────────
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # ── Value head ────────────────────────────────────────────────────────
        # Global average pooling collapses 8×8 spatial dims → 1×1
        # So we go from (B, num_filters, 8, 8) → (B, num_filters)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.value_head = nn.Sequential(
            nn.Linear(num_filters, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)                        # (B, F, 8, 8)
        x = self.residual_tower(x)              # (B, F, 8, 8)
        x = self.global_pool(x)                 # (B, F, 1, 1)
        x = x.view(x.size(0), -1)              # (B, F)
        x = self.value_head(x)                  # (B, 1)
        return torch.tanh(x)                    # [-1, +1]


# ── Keep old classes for backwards compatibility ───────────────────────────────
class SimpleChessNet(nn.Module):
    """
    Original 3-layer CNN — kept for loading old .pth checkpoints.
    New training should use ChessNet instead.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.value_fc1 = nn.Linear(256 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        value = x.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_out(value))
        return value


class PositionEvaluator(nn.Module):
    """Original MLP evaluator — kept for backwards compatibility."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.output(x))
        return x


# ── Test ───────────────────────────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    dummy = torch.randn(4, 13, 8, 8)

    old = SimpleChessNet()
    new = ChessNet()

    old_out = old(dummy)
    new_out = new(dummy)

    print("── SimpleChessNet (old) ──")
    print(f"  Parameters : {count_parameters(old):,}")
    print(f"  Output shape: {old_out.shape}")
    print(f"  Sample eval : {old_out[0].item():.4f}")

    print("\n── ChessNet (new) ──")
    print(f"  Parameters : {count_parameters(new):,}")
    print(f"  Output shape: {new_out.shape}")
    print(f"  Sample eval : {new_out[0].item():.4f}")

if __name__ == "__main__":
    test_model()