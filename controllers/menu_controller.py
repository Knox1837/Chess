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
                self.start_game(use_trained_ai=False)
            
            elif choice == 4:
                self.train_ai_screen()
            
            elif choice == 5:
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
        model_path = "chess_ai_final.pth"
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