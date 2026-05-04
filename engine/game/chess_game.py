"""
Main chess game logic
"""
import chess
import pygame
from engine.game.piece_movement import ChessMovement
from engine.ai.chess_ai import ChessAI
from ui.game_ui import load_piece_images, draw_board, draw_pieces, draw_status, draw_highlights
from utils.config import *

class ChessGame:
    """Main chess game logic"""
    def __init__(self, ai_model_path=None):
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.piece_images = load_piece_images()
        
        # Movement handler
        self.movement = ChessMovement(self.board, SQUARE_SIZE, WIDTH, HEIGHT)
        
        # AI setup
        self.ai = ChessAI(model_path=ai_model_path, skill_level=4 if ai_model_path else 2)
        self.ai_enabled = False
        self.ai_color = chess.BLACK
        self.ai_thinking = False
        
        self.board_flipped = False  # Track if board is to be flipped
        # Highlight surfaces
        self.highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        self.highlight_surf.fill((255, 255, 0, 100))
        
        self.valid_move_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        self.valid_move_surf.fill((0, 255, 0, 100))

        self.last_move_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        self.last_move_surf.fill((255, 255, 0, 80))  # yellow, slightly transparent
    
    def handle_click(self, pos):
        """Handle mouse click for piece movement"""
        return self.movement.handle_click(pos)
    
    def handle_mouse_down(self, pos):
        """Handle mouse down for dragging"""
        return self.movement.handle_mouse_down(pos)
    
    def handle_mouse_up(self, pos):
        """Handle mouse up for dragging"""
        return self.movement.handle_mouse_up(pos)
    
    def handle_mouse_motion(self, pos):
        """Handle mouse motion for dragging"""
        return self.movement.handle_mouse_motion(pos)
    
    def handle_keydown(self, event):
        """Handle keyboard input"""
        return self.movement.handle_keydown(event)
    
    def draw(self, win):
        """Draw the game"""
        draw_board(win)

        # Draw last move highlight
        if self.board.move_stack:
            last_move = self.board.peek()
            for square in [last_move.from_square, last_move.to_square]:
                row = 7 - chess.square_rank(square) if not self.board_flipped else chess.square_rank(square)
                col = chess.square_file(square) if not self.board_flipped else 7 - chess.square_file(square)
                win.blit(self.last_move_surf, (col * SQUARE_SIZE, row * SQUARE_SIZE))

        # Draw highlights
        if self.movement.selected_square is not None:
            draw_highlights(win, self.movement.selected_square, self.movement.valid_moves, SQUARE_SIZE, self.board_flipped)

        # Draw pieces
        draw_pieces(win, self.board, self.piece_images, self.board_flipped)
        
        # Draw dragged piece if any
        self.movement.draw_dragged_piece(win, self.piece_images)
        
        # Draw status
        status_text = self.get_game_status()
        draw_status(win, status_text)
    
    def get_game_status(self):
        """Get current game status"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            return "Stalemate! Draw!"
        elif self.board.is_check():
            return "Check!"
        else:
            return ""
    
    def update(self):
        """Update game state - called each frame"""
        # Update selected square and valid moves from movement handler
        self.selected_square = self.movement.selected_square
        self.valid_moves = self.movement.valid_moves
        
        # AI move if enabled
        if (self.ai_enabled and not self.board.is_game_over() and 
            self.board.turn == self.ai_color and not self.ai_thinking):
            
            self.ai_thinking = True
            
            # Get AI move
            move_result = self.ai.choose_move(self.board)
            
            if move_result:
                self.board.push(move_result['move'])
                print(f"AI plays: {move_result['move_san']} (eval: {move_result['eval']:.2f})")
            
            self.ai_thinking = False
            
            # Clear selection
            self.movement.selected_square = None
            self.movement.valid_moves = []
            
            return True
        
        return False