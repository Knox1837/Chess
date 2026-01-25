import os
import pygame
import chess
from piece_movement import ChessMovement

# ------------------ CONFIG ------------------
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)

ASSETS_DIR = "assets"

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Board")
CLOCK = pygame.time.Clock()

def load_piece_images():
    """Load images for chess pieces and scale them to fit squares"""
    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors = ["w", "b"]
    images = {}
    
    for piece in pieces:
        for color in colors:
            path = os.path.join(ASSETS_DIR, f"{color}-{piece}.png") 
            image = pygame.image.load(path)
            image = pygame.transform.smoothscale(image, (SQUARE_SIZE, SQUARE_SIZE))
            
            if piece == "knight":
                key = f"N{color}"
            else:
                key = f"{piece[0].upper() if color == 'w' else piece[0].lower()}{color}"
            images[key] = image
    return images

PIECE_IMAGES = load_piece_images()

def draw_board(win):
    """Draw the chess board squares"""
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(win, board):
    """Draw chess pieces on the board"""
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
            
            if key in PIECE_IMAGES:
                win.blit(PIECE_IMAGES[key], (col * SQUARE_SIZE, row * SQUARE_SIZE))

def draw_status(win, text):
    """Draw game status text"""
    font = pygame.font.SysFont(None, 30)
    text_surface = font.render(text, True, (0, 0, 0))
    win.blit(text_surface, (10, 10))

def main():
    board = chess.Board()
    movement = ChessMovement(board, SQUARE_SIZE, WIDTH, HEIGHT)
    
    run = True
    while run:
        CLOCK.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    movement.handle_mouse_down(event.pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    movement.handle_mouse_up(event.pos)
            
            elif event.type == pygame.MOUSEMOTION:
                movement.handle_mouse_motion(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                movement.handle_keydown(event)
        
        # Draw everything
        draw_board(WIN)
        movement.draw_highlights(WIN)
        draw_pieces(WIN, board)
        movement.draw_dragged_piece(WIN, PIECE_IMAGES)
        
        status_text = movement.get_game_status()
        draw_status(WIN, status_text)
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()