"""
Main entry point for Chess AI System
"""
import os
import sys
import pygame
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import local modules
from utils.config import *
from controllers.menu_controller import MenuController
from controllers.game_controller import GameController

def main():
    """Main program entry point"""
    # Setup pygame
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess AI Trainer & Player")
    CLOCK = pygame.time.Clock()
    
    # Check command line arguments
    parser = argparse.ArgumentParser(description='Chess AI System')
    parser.add_argument('--train', action='store_true', help='Train AI immediately')
    parser.add_argument('--play', action='store_true', help='Play immediately')
    parser.add_argument('--pgn', type=str, help='PGN file to process')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to process')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    
    args = parser.parse_args()
    
    if args.train:
        # Train immediately
        train_immediately(args)
    elif args.play:
        # Play immediately
        play_immediately(WIN, CLOCK)
    else:
        # Show menu
        show_menu_interface(WIN, CLOCK)

def train_immediately(args):
    """Handle --train command line argument"""
    try:
        from engine.data.pgn_processor import LichessPGNProcessor as PGNProcessor
        from engine.ai.chess_trainer import ChessTrainer
    except ImportError as e:
        print(f"Error: Training modules not available: {e}")
        sys.exit(1)
    
    # Find PGN file
    if args.pgn:
        pgn_path = args.pgn
    else:
        # Look for PGN in engine/data/raw
        pgn_files = []
        if os.path.exists("engine/data/raw"):
            pgn_files = [f for f in os.listdir("engine/data/raw") if f.endswith('.pgn')]
        
        if pgn_files:
            pgn_path = os.path.join("engine/data/raw", pgn_files[0])
        else:
            print("No PGN file found. Please place a .pgn file in engine/data/raw/")
            sys.exit(1)
    
    print(f"Training AI with {args.games} games from {pgn_path}...")
    processor = PGNProcessor()
    processor.process_pgn_file(pgn_path, max_games=args.games)
    
    trainer = ChessTrainer(model_type="simple")
    trainer.train(num_epochs=args.epochs)
    
    print("\n✅ Training complete! You can now run: python main.py --play")

def play_immediately(win, clock):
    """Handle --play command line argument"""
    menu_controller = MenuController(win, clock)
    menu_controller.quick_start()

def show_menu_interface(win, clock):
    """Show the main menu interface"""
    menu_controller = MenuController(win, clock)
    menu_controller.run()

if __name__ == "__main__":
    main()