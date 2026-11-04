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

def install_fullscreen_hotkey():
    """Make F11 toggle fullscreen from any screen (menu or game)."""
    original_get = pygame.event.get

    def get(*args, **kwargs):
        events = original_get(*args, **kwargs)
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                try:
                    pygame.display.toggle_fullscreen()
                except pygame.error as e:
                    print(f"Could not toggle fullscreen: {e}")
        return events

    pygame.event.get = get


def maximize_window():
    """Maximize the game window (best effort; returns True on success)."""
    try:
        from pygame._sdl2.video import Window
        Window.from_display_module().maximize()
        return True
    except Exception:
        pass
    if sys.platform == "win32":                      # fallback: plain Win32 call
        try:
            import ctypes
            hwnd = pygame.display.get_wm_info()["window"]
            ctypes.windll.user32.ShowWindow(hwnd, 3)  # 3 = SW_MAXIMIZE
            return True
        except Exception:
            pass
    return False


def main():
    """Main program entry point"""
    # Smooth (instead of blocky) scaling when the window is enlarged
    os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "linear")

    # Setup pygame
    pygame.init()
    # RESIZABLE lets you maximize / drag-resize the window; SCALED keeps the 600x600 game surface and stretches it to the window (mouse positions are converted automatically, so no layout changes).
    flags = pygame.RESIZABLE | getattr(pygame, "SCALED", 0)
    WIN = pygame.display.set_mode((WIDTH, HEIGHT), flags)
    if not WIN.get_flags() & getattr(pygame, "SCALED", 0):
        print("Note: scaled window mode is not available here, so a maximized window will not stretch the board.")
    install_fullscreen_hotkey()
    pygame.display.set_caption("Chess AI Trainer & Player")
    CLOCK = pygame.time.Clock()
    
    # Check command line arguments
    parser = argparse.ArgumentParser(description='Chess AI System')
    parser.add_argument('--train', action='store_true', help='Train AI immediately')
    parser.add_argument('--play', action='store_true', help='Play immediately')
    parser.add_argument('--pgn', type=str, help='PGN file to process')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to process')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--windowed', action='store_true', help='Start in a normal window instead of maximized')
    
    args = parser.parse_args()

    if not args.windowed and not maximize_window():
        print("Note: could not maximize the window automatically; use the maximize button or F11.")
    
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
    
    print("\nTraining complete! You can now run: python main.py --play")

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