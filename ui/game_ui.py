"""
UI drawing functions
"""
import pygame
import chess
from utils.config import *

def load_piece_images():
    """Load chess piece images"""
    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors = ["w", "b"]
    images = {}
    
    for piece in pieces:
        for color in colors:
            try:
                import os
                path = os.path.join(ASSETS_DIR, f"{color}-{piece}.png")
                image = pygame.image.load(path)
                image = pygame.transform.smoothscale(image, (SQUARE_SIZE, SQUARE_SIZE))
                
                if piece == "knight":
                    key = f"N{color}"
                else:
                    key = f"{piece[0].upper() if color == 'w' else piece[0].lower()}{color}"
                images[key] = image
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                # Create placeholder
                surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                surf.fill((255, 0, 0) if color == 'w' else (0, 0, 255))
                images[key] = surf
    
    return images

def draw_board(win):
    """Draw chess board"""
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(win, board, piece_images):
    """Draw chess pieces"""
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            
            symbol = piece.symbol()
            color = 'w' if piece.color == chess.WHITE else 'b'
            if symbol.lower() == "n":
                key = f"N{color}"
            else:
                key = f"{symbol.upper() if color == 'w' else symbol.lower()}{color}"
            
            if key in piece_images:
                win.blit(piece_images[key], (col * SQUARE_SIZE, row * SQUARE_SIZE))

def draw_status(win, text):
    """Draw game status"""
    if text:
        font = pygame.font.SysFont(None, 30)
        text_surface = font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
        win.blit(text_surface, text_rect)

def draw_highlights(win, selected_square, valid_moves, square_size):
    """Draw square highlights for selected piece and valid moves"""
    if selected_square is not None:
        # Create highlight surfaces
        highlight_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        highlight_surf.fill((255, 255, 0, 100))
        
        valid_move_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        valid_move_surf.fill((0, 255, 0, 100))
        
        # Draw selected square
        row = 7 - chess.square_rank(selected_square)
        col = chess.square_file(selected_square)
        win.blit(highlight_surf, (col * square_size, row * square_size))
        
        # Draw valid moves
        for move in valid_moves:
            row = 7 - chess.square_rank(move.to_square)
            col = chess.square_file(move.to_square)
            win.blit(valid_move_surf, (col * square_size, row * square_size))