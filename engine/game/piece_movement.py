"""
Piece movement and dragging logic
"""
import pygame
import chess
from utils.config import *

class ChessMovement:
    """Handles piece movement and dragging"""
    def __init__(self, board, square_size, width, height):
        self.board = board
        self.square_size = square_size
        self.width = width
        self.height = height
        self.selected_square = None
        self.dragged_piece = None
        self.drag_pos = None
        self.valid_moves = []
    
    def get_square_from_mouse(self, pos):
        """Convert mouse position to chess square"""
        x, y = pos
        col = x // self.square_size
        row = 7 - (y // self.square_size)
        
        if 0 <= row < 8 and 0 <= col < 8:
            return chess.square(col, row)
        return None
    
    def handle_mouse_down(self, pos):
        """Handle mouse down event for dragging"""
        square = self.get_square_from_mouse(pos)
        if square is not None:
            piece = self.board.piece_at(square)
            if piece and ((self.board.turn == chess.WHITE and piece.color == chess.WHITE) or
                         (self.board.turn == chess.BLACK and piece.color == chess.BLACK)):
                self.selected_square = square
                self.dragged_piece = piece
                self.drag_pos = pos
                self.valid_moves = [move for move in self.board.legal_moves 
                                  if move.from_square == square]
                return True
        return False
    
    def handle_mouse_up(self, pos):
        """Handle mouse up event (for dragging)"""
        if self.dragged_piece and self.selected_square is not None:
            to_square = self.get_square_from_mouse(pos)
            if to_square is not None:
                for move in self.valid_moves:
                    if move.to_square == to_square:
                        self.board.push(move)
                        break
            self.dragged_piece = None
            self.drag_pos = None
            self.selected_square = None
            self.valid_moves = []
        return False
    
    def handle_click(self, pos):
        """Handle click (alternative to drag)"""
        square = self.get_square_from_mouse(pos)
        if square is None:
            return False
        
        piece = self.board.piece_at(square)
        
        # If a square is already selected
        if self.selected_square is not None:
            # Clicking same square deselects
            if square == self.selected_square:
                self.selected_square = None
                self.valid_moves = []
                return True
            
            # Try to move to clicked square
            for move in self.valid_moves:
                if move.to_square == square:
                    self.board.push(move)
                    self.selected_square = None
                    self.valid_moves = []
                    return True
            
            # If clicked a different piece, select it
            if piece:
                self.selected_square = square
                self.valid_moves = [move for move in self.board.legal_moves 
                                  if move.from_square == square]
                return True
            
            # Clicked empty square - deselect
            self.selected_square = None
            self.valid_moves = []
            return False
        
        # No square selected yet - try to select a piece
        if piece and ((self.board.turn == chess.WHITE and piece.color == chess.WHITE) or
                     (self.board.turn == chess.BLACK and piece.color == chess.BLACK)):
            self.selected_square = square
            self.valid_moves = [move for move in self.board.legal_moves 
                              if move.from_square == square]
            return True
        
        return False
    
    def handle_mouse_motion(self, pos):
        """Update drag position"""
        if self.dragged_piece:
            self.drag_pos = pos
    
    def handle_keydown(self, event):
        """Handle keyboard input"""
        if event.key == pygame.K_r:
            self.board.reset()
            self.selected_square = None
            self.valid_moves = []
            print("Board reset")
        elif event.key == pygame.K_z:
            if self.board.move_stack:
                self.board.pop()
                self.selected_square = None
                self.valid_moves = []
                print("Undo move")
    
    def draw_dragged_piece(self, win, piece_images):
        """Draw piece being dragged"""
        if self.dragged_piece and self.drag_pos:
            symbol = self.dragged_piece.symbol()
            color = 'w' if self.dragged_piece.color == chess.WHITE else 'b'
            if symbol.lower() == "n":
                key = f"N{color}"
            else:
                key = f"{symbol.upper() if color == 'w' else symbol.lower()}{color}"
            
            if key in piece_images:
                # Draw at mouse position
                x, y = self.drag_pos
                win.blit(piece_images[key], (x - self.square_size//2, y - self.square_size//2))