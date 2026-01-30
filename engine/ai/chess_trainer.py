"""
AI Trainer
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from engine.models.chess_net import SimpleChessNet, PositionEvaluator
except ImportError:
    SimpleChessNet = None
    PositionEvaluator = None

class ChessDataset(Dataset):
    """Dataset for chess positions"""
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        
        # Load positions and results
        self.positions = np.load(data_dir / "positions.npy")
        self.results = np.load(data_dir / "results.npy")
        
        print(f"Loaded dataset from {data_dir}")
        print(f"Positions: {self.positions.shape}")
        print(f"Results: {self.results.shape}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.positions[idx], dtype=torch.float32),
            torch.tensor(self.results[idx], dtype=torch.float32)
        )

class ChessTrainer:
    def __init__(self, model_type="simple", device=None):
        if SimpleChessNet is None:
            raise ImportError("Neural network models not available")
        
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Initialize model
        if model_type == "simple":
            self.model = SimpleChessNet().to(self.device)
        else:
            self.model = PositionEvaluator().to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for positions, results in loader:
            positions = positions.to(self.device)
            results = results.to(self.device)

            # Handle both tuple output and single output
            model_output = self.model(positions)
            
            # If model returns tuple (value, policy), extract value
            if isinstance(model_output, tuple):
                preds = model_output[0]  # Take value from tuple
            else:
                preds = model_output
                
            preds = preds.squeeze()
            loss = self.criterion(preds, results)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for positions, results in loader:
                positions = positions.to(self.device)
                results = results.to(self.device)

                # Handle both tuple output and single output
                model_output = self.model(positions)
                
                # If model returns tuple (value, policy), extract value
                if isinstance(model_output, tuple):
                    preds = model_output[0]  # Take value from tuple
                else:
                    preds = model_output
                    
                preds = preds.squeeze()
                loss = self.criterion(preds, results)
                total_loss += loss.item()

        return total_loss / len(loader)
    
    def train(self, num_epochs=20, batch_size=64, data_dir=None):
        """Main training loop"""
        # Load data
        if data_dir is None:
            # Find the latest processed data
            processed_dir = Path("engine/data/processed")
            checkpoints = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
            if checkpoints:
                data_dir = checkpoints[-1]
            else:
                print("Error: No processed data found!")
                print("Please run the PGN processor first")
                return
        
        dataset = ChessDataset(data_dir)
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nTraining Details:")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth")
                print(f"  Saved best model (val loss: {val_loss:.4f})")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save final model
        self.save_model("final_model.pth")
        self.plot_training_history()
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, filename):
        """Save model to file"""
        model_dir = Path("engine/models/saved")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'model_class': self.model.__class__.__name__
        }, save_path)
        
        print(f"  Model saved to: {save_path}")
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Training & Validation Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        axes[0].fill_between(epochs, self.history['train_loss'], self.history['val_loss'], 
                            alpha=0.2, color='gray')
        axes[0].set_title('Training & Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Learning Rate
        axes[1].plot(epochs, self.history['lr'], 'g-', linewidth=2, marker='o', markersize=4)
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_yscale('log')  # Log scale for LR often shows changes better
        axes[1].grid(True, alpha=0.3)
        
        # 3. Loss Difference
        loss_diff = [t - v for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        axes[2].plot(epochs, loss_diff, color='purple', linewidth=2, linestyle='--', marker='s', markersize=4)
        axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        axes[2].fill_between(epochs, loss_diff, 0, where=[ld > 0 for ld in loss_diff], 
                            alpha=0.3, color='green', interpolate=True)
        axes[2].fill_between(epochs, loss_diff, 0, where=[ld < 0 for ld in loss_diff], 
                            alpha=0.3, color='red', interpolate=True)
        axes[2].set_title('Train-Val Loss Difference')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss Difference')
        axes[2].grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Chess AI Training History (Best Val Loss: {min(self.history["val_loss"]):.4f})', 
                    fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        # Save figure
        save_dir = Path("engine/models/saved")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save multiple versions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(save_dir / f"training_history_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.savefig(save_dir / "training_history_latest.png", dpi=150, bbox_inches='tight')
        
        plt.show()