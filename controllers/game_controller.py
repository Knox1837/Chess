"""
Game Controller - handles the main game loop and events
"""
import pygame
import chess
import os

class GameController:
    """Controls the chess game loop and user input"""
    def __init__(self, win, clock, use_trained_ai=False, human_color=chess.WHITE):
        from engine.game.chess_game import ChessGame
        
        self.win = win
        self.clock = clock
        
        # use_trained_ai now just selects the strong AI setting (skill level 4); no model file is needed
        self.game = ChessGame(strong_ai=use_trained_ai, human_color=human_color)
        
        # Enable AI if requested
        if use_trained_ai:
            self.game.ai_enabled = True
        
    def handle_events(self):
        """Process all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if self._handle_keydown(event) is False:      # ESC
                return False
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
                self.game.switch_sides()
            
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
        
        # Stop the AI's background thinking and close its engine process
        self.game.ai.close()
        return True