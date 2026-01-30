"""
Game Controller - handles the main game loop and events
"""
import pygame
import chess
import os

class GameController:
    """Controls the chess game loop and user input"""
    
    def __init__(self, win, clock, use_trained_ai=False):
        from engine.game.chess_game import ChessGame
        
        self.win = win
        self.clock = clock
        
        # Load appropriate AI
        if use_trained_ai:
            model_path = "chess_ai_final.pth"
            if not os.path.exists(model_path):
                print("No trained model found. Using basic AI instead.")
                model_path = None
        else:
            model_path = None
        
        # Create game instance
        self.game = ChessGame(ai_model_path=model_path)
        
    def handle_events(self):
        """Process all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            self._handle_keydown(event)
            self._handle_mouse_events(event)
        
        return True
    
    def _handle_keydown(self, event):
        """Handle keyboard input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            
            # AI controls
            elif event.key == pygame.K_a:
                self.game.ai_enabled = not self.game.ai_enabled
                status = "enabled" if self.game.ai_enabled else "disabled"
                print(f"AI {status}")
            
            elif event.key == pygame.K_s:
                self.game.ai_color = chess.WHITE if self.game.ai_color == chess.BLACK else chess.BLACK
                color = "White" if self.game.ai_color == chess.WHITE else "Black"
                print(f"AI now plays {color}")
            
            elif event.key == pygame.K_r:
                self.game.board.reset()
                self.game.movement.selected_square = None
                self.game.movement.valid_moves = []
                print("Board reset")
            
            elif event.key == pygame.K_z:
                if self.game.board.move_stack:
                    self.game.board.pop()
                    self.game.movement.selected_square = None
                    self.game.movement.valid_moves = []
                    print("Undo move")
        
        return True
    
    def _handle_mouse_events(self, event):
        """Handle mouse events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.game.handle_mouse_down(event.pos)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.game.handle_mouse_up(event.pos)
                # Also handle as click
                self.game.handle_click(event.pos)
        
        elif event.type == pygame.MOUSEMOTION:
            self.game.handle_mouse_motion(event.pos)
    
    def update(self):
        """Update game state"""
        return self.game.update()
    
    def draw(self):
        """Draw everything to the screen"""
        self.win.fill((0, 0, 0))
        self.game.draw(self.win)
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update game state
            self.update()
            
            # Draw everything
            self.draw()
            
            # Control frame rate
            self.clock.tick(60)
        
        return True