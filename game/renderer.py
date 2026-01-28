import pygame
import chess
import os

class ChessRenderer:
    """Class to handle rendering of the chess game using Pygame"""

    WIDTH, HEIGHT = 600, 600 # default window size, this cant be changed with initialization of renderer object
    SQUARE_SIZE = WIDTH // 8
    WHITE = (240, 217, 181)
    BLACK = (181, 136, 99)
    ASSETS_DIR = "assets"
    
    @staticmethod
    def load_piece_images():
        """Load chess piece images"""
        pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
        colors = ["w", "b"]
        images = {}
        
        for piece in pieces:
            for color in colors:
                try:
                    path = os.path.join(ChessRenderer.ASSETS_DIR, f"{color}-{piece}.png")
                    image = pygame.image.load(path)
                    image = pygame.transform.smoothscale(image, (ChessRenderer.SQUARE_SIZE, ChessRenderer.SQUARE_SIZE))
                    
                    if piece == "knight":
                        key = f"N{color}"
                    else:
                        key = f"{piece[0].upper() if color == 'w' else piece[0].lower()}{color}"
                    images[key] = image
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
                    # Create placeholder
                    surf = pygame.Surface((ChessRenderer.SQUARE_SIZE, ChessRenderer.SQUARE_SIZE))
                    surf.fill((255, 0, 0) if color == 'w' else (0, 0, 255))
                    images[key] = surf
        
        return images
    @staticmethod
    def draw_board(win):
        """Draw chess board"""
        for row in range(8):
            for col in range(8):
                color = ChessRenderer.WHITE if (row + col) % 2 == 0 else ChessRenderer.BLACK
                pygame.draw.rect(win, color, (col * ChessRenderer.SQUARE_SIZE, row * ChessRenderer.SQUARE_SIZE, ChessRenderer.SQUARE_SIZE, ChessRenderer.SQUARE_SIZE))

    @staticmethod
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
                    win.blit(piece_images[key], (col * ChessRenderer.SQUARE_SIZE, row * ChessRenderer.SQUARE_SIZE))

    @staticmethod
    def draw_status(win, text):
        """Draw game status"""
        if text:
            font = pygame.font.SysFont(None, 30)
            text_surface = font.render(text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(ChessRenderer.WIDTH//2, ChessRenderer.HEIGHT//2))
            win.blit(text_surface, text_rect)

    # def draw_controls(win, ai_enabled, ai_color, ai_thinking):
    #     """Draw control information"""
    #     font = pygame.font.SysFont(None, 24)
        
    #     # AI status
    #     if ai_enabled:
    #         status = f"AI ({'White' if ai_color == chess.WHITE else 'Black'})"
    #         if ai_thinking:
    #             status += " - Thinking..."
    #         ai_surface = font.render(status, True, (255, 0, 0))
    #         win.blit(ai_surface, (WIDTH - 200, 10))
        
    #     # Controls
    #     controls = [
    #         "CONTROLS:",
    #         "A - Toggle AI opponent",
    #         "S - Switch AI side",
    #         "R - Reset board",
    #         "Z - Undo move",
    #         "ESC - Return to menu"
    #     ]
        
    #     for i, text in enumerate(controls):
    #         surface = font.render(text, True, (50, 50, 50))
    #         win.blit(surface, (10, 10 + i * 25))
