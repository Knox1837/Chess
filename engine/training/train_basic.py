"""
Basic training script for chess neural network
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import random

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


class ChessAITrainer:
    """Train the neural network"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = SimpleChessNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, epochs=10, batch_size=32):
        """Train the model"""
        # Load data
        positions = np.load("processed_data/positions.npy")
        results = np.load("processed_data/results.npy")
        
        # Create dataset
        dataset = list(zip(positions, results))
        random.shuffle(dataset)
        
        # Split train/validation
        split = int(0.8 * len(dataset))
        train_data = dataset[:split]
        val_data = dataset[split:]
        
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            random.shuffle(train_data)
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                batch_positions = torch.tensor([p for p, _ in batch], dtype=torch.float32).to(self.device)
                batch_results = torch.tensor([r for _, r in batch], dtype=torch.float32).to(self.device)
                
                # Forward pass
                predictions = self.model(batch_positions).squeeze()
                loss = self.criterion(predictions, batch_results)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / (len(train_data) / batch_size)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i+batch_size]
                    batch_positions = torch.tensor([p for p, _ in batch], dtype=torch.float32).to(self.device)
                    batch_results = torch.tensor([r for _, r in batch], dtype=torch.float32).to(self.device)
                    
                    predictions = self.model(batch_positions).squeeze()
                    loss = self.criterion(predictions, batch_results)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / (len(val_data) / batch_size)
            self.history['val_loss'].append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save model checkpoint
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.save_model(f"chess_ai_epoch_{epoch+1}.pth")
        
        # Plot training history
        self.plot_training_history()
        
        print(f"\nTraining complete! Model saved as 'chess_ai_final.pth'")
        self.save_model("chess_ai_final.pth")
    
    def save_model(self, filename):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filename)
        print(f"  Saved model to {filename}")
    
    def plot_training_history(self):
        """Plot training metrics"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
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
    trainer = ChessAITrainer(model_type="simple")
    trainer.train(num_epochs=20, batch_size=64)

if __name__ == "__main__":
    main()