"""
Menu Controller - handles menu navigation
"""
import pygame
import sys
import os
from pathlib import Path

class MenuController:
    """Controls menu navigation and training"""
    
    def __init__(self, win, clock):
        from ui.menu import show_menu, train_ai_screen
        from controllers.game_controller import GameController
        
        self.win = win
        self.clock = clock
        self.show_menu = show_menu
        self.train_ai_screen = train_ai_screen
        self.GameController = GameController
    
    def run(self):
        """Main menu loop"""
        print("="*60)
        print("CHESS AI TRAINER & PLAYER")
        print("="*60)
        print("\nInstructions:")
        print("1. Place PGN files in: engine/data/raw/")
        print("2. Train AI first (Option 4)")
        print("3. Play against trained AI (Option 2)")
        print("4. Play vs Stockfish (Option 5)")
        print("5. Exit (Option 6 or ESC)")
        print("="*60)
        
        while True:
            choice = self.show_menu(self.win, self.clock)
            
            if choice == 1:
                print("\nStarting Human vs Human game...")
                self.start_game(use_trained_ai=False)
            
            elif choice == 2:
                print("\nStarting game vs Trained AI...")
                self.start_game(use_trained_ai=True)
            
            elif choice == 3:
                print("\nStarting game vs Basic AI...")
                game = self.GameController(self.win, self.clock, use_trained_ai=False)
                game.game.ai_enabled = True
                game.run()
            
            elif choice == 4:
                self.train_ai_screen()
            
            elif choice ==5:
                self.start_stockfish_game()

            elif choice == 6:
                print("\nThanks for playing!")
                break
        
        pygame.quit()
        sys.exit()
    
    def start_game(self, use_trained_ai=False):
        """Start a new game"""
        game = self.GameController(self.win, self.clock, use_trained_ai)
        game.run()
    
    def quick_start(self):
        """Quick start for immediate play"""
        print("Quick Start: Loading chess game...")
        
        # Try to load trained model
        model_path = os.path.join("engine", "models", "saved", "chess_ai_final.pth")
        if not os.path.exists(model_path):
            print("No trained AI found. Using basic AI.")
            model_path = None
        
        game = self.GameController(self.win, self.clock, model_path is not None)
        game.game.ai_enabled = True
        
        print("\nGame Started!")
        print("Controls:")
        print("  Click/Drag - Select and move pieces")
        print("  A - Toggle AI")
        print("  S - Switch AI side")
        print("  R - Reset board")
        print("  Z - Undo move")
        print("  ESC - Quit")
        print("\n" + "="*50)
        
        game.run()
    
    def start_stockfish_game(self):
        """Play directly against Stockfish engine"""
        import chess.engine
        from pathlib import Path

        stockfish_path = Path("engine/stockfish.exe")
        if not stockfish_path.exists():
            print("Stockfish not found at engine/stockfish.exe")
            print("Please download from https://stockfishchess.org/download/")
            input("Press Enter to return to menu...")
            return

        game = self.GameController(self.win, self.clock, use_trained_ai=False)
        game.game.ai_enabled = True

        # Patch the AI's choose_move to use Stockfish directly
        engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))

        original_choose_move = game.game.ai.choose_move

        def stockfish_move(board):
            try:
                result = engine.play(board, chess.engine.Limit(time=1.0))
                return {
                    'move': result.move,
                    'eval': 0,
                    'move_san': board.san(result.move)
                }
            except Exception as e:
                print(f"Stockfish error: {e}")
                return original_choose_move(board)

        game.game.ai.choose_move = stockfish_move

        print("Playing vs Stockfish (1 second per move)")
        print("Controls: A=toggle AI, S=switch sides, R=reset, Z=undo, ESC=menu")

        game.run()
        engine.quit()