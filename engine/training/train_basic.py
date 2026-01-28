"""
Basic training script for chess neural network
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
from models.chess_net import SimpleChessNet, PositionEvaluator

class ChessDataset(Dataset):
    """Dataset for chess positions"""
    def __init__(self, data_dir="engine/data/processed/checkpoint_100"):
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
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (positions, results) in enumerate(pbar):
            positions = positions.to(self.device)
            results = results.to(self.device)
            
            # Forward pass
            if isinstance(self.model, SimpleChessNet):
                values, _ = self.model(positions)
                predictions = values.squeeze()
            else:
                predictions = self.model(positions).squeeze()
            
            # Calculate loss
            loss = self.criterion(predictions, results)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for positions, results in val_loader:
                positions = positions.to(self.device)
                results = results.to(self.device)
                
                if isinstance(self.model, SimpleChessNet):
                    values, _ = self.model(positions)
                    predictions = values.squeeze()
                else:
                    predictions = self.model(positions).squeeze()
                
                loss = self.criterion(predictions, results)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
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
        save_dir = Path("engine/models/saved")
        save_dir.mkdir(exist_ok=True)
        
        save_path = save_dir / filename
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
        
        # Training loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate
        axes[1].plot(epochs, self.history['lr'], 'g-')
        axes[1].set_title('Learning Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(True)
        
        # Loss difference
        loss_diff = [t - v for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        axes[2].plot(epochs, loss_diff, 'purple-')
        axes[2].set_title('Train-Val Loss Difference')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss Difference')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        save_dir = Path("engine/models/saved")
        plt.savefig(save_dir / "training_history.png", dpi=100)
        plt.show()

def main():
    """Main training function"""
    print("Starting Chess AI Training")
    print("=" * 60)
    
    # Check if processed data exists
    processed_dir = Path("engine/data/processed")
    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        print("Error: No processed data found!")
        print("\nPlease run the PGN processor first:")
        print("python engine/data/pgn_processor.py --pgn path/to/your.pgn --max-games 1000")
        print("\nOr copy your PGN file to: engine/data/raw/ and run:")
        print("python engine/setup.py --process-pgn")
        return
    
    # Start training
    trainer = ChessTrainer(model_type="simple")
    trainer.train(num_epochs=20, batch_size=64)

if __name__ == "__main__":
    main()