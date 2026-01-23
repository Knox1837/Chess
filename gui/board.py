import os
import pygame
import chess

# ------------------ CONFIG ------------------
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS # 75x75 pixels

# Colors
WHITE = (240, 217, 181) # Light-ish square color
BLACK = (181, 136, 99)  # brownish dark square color

ASSETS_DIR = "assets"
#ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Board")
CLOCK = pygame.time.Clock() #for controlling frame rate

def load_piece_images():
    """load images for chess pieces and scale them to fit squares"""
    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors = ["w", "b"] #nameing scheme: color-piece
    images = {}

    for piece in pieces:
        for color in colors:
            path = os.path.join(ASSETS_DIR, f"{color}-{piece}.png") 
            image = pygame.image.load(path)
            image = pygame.transform.smoothscale(image, (SQUARE_SIZE, SQUARE_SIZE)) # scale to fit square 75x75

            if piece == "knight":
                key = f"N{color}"  # knight = N to avoid confusion with king K in FEN notation e.g: Nw, Nb
            else:
                key = f"{piece[0].upper() if color == 'w' else piece[0].lower()}{color}" #every other piece uses first letter e.g: Pw, Bw
            images[key] = image # loads images into dictionary corresponding to FEN notation keys
    return images

PIECE_IMAGES = load_piece_images()

def draw_board(win):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BLACK # alternating colors starting with black at bottom-left
            pygame.draw.rect(win, color, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# ------------------ DRAW PIECES ------------------
def draw_pieces(win, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square) #Gets the piece object at a given square on the board
        if piece:
            row = 7 - chess.square_rank(square) # Pygame's y=0 is at the top, chess library's rank 0 is at bottom so 0,0 is a1 bottom left
            col = chess.square_file(square) 

            # Map FEN symbol to PIECE_IMAGES key
            symbol = piece.symbol() # Returns the FEN symbol of a piece as a string.
            color = 'w' if piece.color == chess.WHITE else 'b'
            if symbol.lower() == "n":
                key = f"N{color}"
            else:
                key = f"{symbol.upper() if color == 'w' else symbol.lower()}{color}"

            if key in PIECE_IMAGES: # Draw the piece image at the correct position
                WIN.blit(PIECE_IMAGES[key], (col*SQUARE_SIZE, row*SQUARE_SIZE))

def main():
    board = chess.Board()  # Start position

    run = True
    while run:
        CLOCK.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        draw_board(WIN)
        draw_pieces(WIN, board)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
