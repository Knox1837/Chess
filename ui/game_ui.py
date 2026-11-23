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

_coord_font = None

def draw_coordinates(win, flipped=False):
    """Draw file letters (bottom edge) and rank numbers (left edge) inside the edge squares"""
    global _coord_font
    if _coord_font is None:
        _coord_font = pygame.font.SysFont("arial", max(12, SQUARE_SIZE // 5), bold=True)
    pad = 3
    for i in range(8):
        # file letter on the bottom row, at that column's bottom-right corner
        file_char = chr(ord('a') + (7 - i if flipped else i))
        light = (7 + i) % 2 == 0                      # square colour at (row 7, col i)
        text = _coord_font.render(file_char, True, BLACK if light else WHITE)
        win.blit(text, ((i + 1) * SQUARE_SIZE - text.get_width() - pad,
                        HEIGHT - text.get_height() - pad))
        # rank number on the left column, at that row's top-left corner
        rank_char = str(i + 1 if flipped else 8 - i)
        light = (i + 0) % 2 == 0                      # square colour at (row i, col 0)
        text = _coord_font.render(rank_char, True, BLACK if light else WHITE)
        win.blit(text, (pad, i * SQUARE_SIZE + pad))

def draw_pieces(win, board, piece_images, flipped=False):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square) if not flipped else chess.square_rank(square)
            col = chess.square_file(square) if not flipped else 7 - chess.square_file(square)
            
            symbol = piece.symbol()
            color = 'w' if piece.color == chess.WHITE else 'b'
            if symbol.lower() == "n":
                key = f"N{color}"
            else:
                key = f"{symbol.upper() if color == 'w' else symbol.lower()}{color}"
            
            if key in piece_images:
                win.blit(piece_images[key], (col * SQUARE_SIZE, row * SQUARE_SIZE))

# move list panel (to the right of the board)
PANEL_BG = (38, 36, 33)
ROW_ALT = (44, 42, 39)
LAST_MOVE_BG = (88, 86, 82)
HEADER_H = 40
ROW_H = 30
ROWS_VISIBLE = (HEIGHT - HEADER_H - 12) // ROW_H
_panel_fonts = {}

def _panel_font(size, bold=False):
    key = (size, bold)
    if key not in _panel_fonts:
        _panel_fonts[key] = pygame.font.SysFont("arial", size, bold=bold)
    return _panel_fonts[key]

def move_list_max_scroll(n_plies, has_result=False):
    """Highest scroll value (first visible row) for a move list with n_plies half-moves"""
    rows = (n_plies + 1) // 2 + (1 if has_result else 0)
    return max(0, rows - ROWS_VISIBLE)

def draw_move_list(win, moves, scroll, result=None):
    """Chess.com-style move list: one row per full move (number, White's move, Black's move).
    moves: SAN strings in game order; scroll: index of the first visible row; result: '1-0', '0-1' or '1/2-1/2'."""
    x0 = WIDTH
    pygame.draw.rect(win, PANEL_BG, (x0, 0, PANEL_WIDTH, HEIGHT))
    pygame.draw.line(win, (20, 20, 20), (x0, 0), (x0, HEIGHT), 2)

    title = _panel_font(22, True).render("Moves", True, (235, 235, 235))
    win.blit(title, (x0 + 14, (HEADER_H - title.get_height()) // 2))
    pygame.draw.line(win, (62, 60, 57), (x0, HEADER_H), (x0 + PANEL_WIDTH, HEADER_H), 1)

    num_w = 46
    col_w = (PANEL_WIDTH - num_w - 14) // 2
    full_rows = (len(moves) + 1) // 2
    total_rows = full_rows + (1 if result else 0)
    num_font, move_font = _panel_font(18), _panel_font(20, True)

    for r in range(ROWS_VISIBLE):
        idx = scroll + r
        if idx >= total_rows:
            break
        y = HEADER_H + 6 + r * ROW_H
        if idx % 2 == 1:
            pygame.draw.rect(win, ROW_ALT, (x0, y, PANEL_WIDTH, ROW_H))
        if idx >= full_rows:                                  # the result line under the last move
            res = move_font.render(result, True, (235, 235, 235))
            win.blit(res, (x0 + (PANEL_WIDTH - res.get_width()) // 2, y + (ROW_H - res.get_height()) // 2))
            continue
        num = num_font.render(f"{idx + 1}.", True, (150, 150, 150))
        win.blit(num, (x0 + num_w - 8 - num.get_width(), y + (ROW_H - num.get_height()) // 2))
        for c in (0, 1):
            ply = idx * 2 + c
            if ply >= len(moves):
                break
            cell = pygame.Rect(x0 + num_w + c * col_w, y + 2, col_w - 4, ROW_H - 4)
            if ply == len(moves) - 1:                         # the latest move is highlighted
                pygame.draw.rect(win, LAST_MOVE_BG, cell, border_radius=4)
            txt = move_font.render(moves[ply], True, (240, 240, 240))
            win.blit(txt, (cell.x + 8, cell.y + (cell.height - txt.get_height()) // 2))

    if total_rows > ROWS_VISIBLE:                             # scrollbar
        track_h = ROWS_VISIBLE * ROW_H
        bar_h = max(24, track_h * ROWS_VISIBLE // total_rows)
        max_scroll = total_rows - ROWS_VISIBLE
        bar_y = HEADER_H + 6 + (track_h - bar_h) * scroll // max(1, max_scroll)
        pygame.draw.rect(win, (100, 98, 94), (x0 + PANEL_WIDTH - 7, bar_y, 4, bar_h), border_radius=2)

def draw_status(win, text, hint=None):
    """Draw the game status (check, checkmate, draws) on a light box in the middle of the board"""
    if not text:
        return
    font = pygame.font.SysFont(None, 34)
    text_surface = font.render(text, True, (0, 0, 0))
    lines = [text_surface]
    if hint:
        lines.append(pygame.font.SysFont(None, 24).render(hint, True, (60, 60, 60)))
    width = max(s.get_width() for s in lines) + 40
    height = sum(s.get_height() for s in lines) + 24 + 6 * (len(lines) - 1)
    box = pygame.Surface((width, height), pygame.SRCALPHA)
    box.fill((255, 255, 255, 225))
    pygame.draw.rect(box, (90, 90, 90), box.get_rect(), 2)
    y = 12
    for s in lines:
        box.blit(s, ((width - s.get_width()) // 2, y))
        y += s.get_height() + 6
    win.blit(box, box.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

def draw_highlights(win, selected_square, valid_moves, square_size, flipped=False):
    """Draw square highlights for selected piece and valid moves"""
    if selected_square is not None:
        # Create highlight surfaces
        highlight_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        highlight_surf.fill((255, 255, 0, 120))
        
        valid_move_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        valid_move_surf.fill((0, 255, 0, 100))
        
        # Draw selected square
        row = 7 - chess.square_rank(selected_square) if not flipped else chess.square_rank(selected_square)
        col = chess.square_file(selected_square) if not flipped else 7 - chess.square_file(selected_square)
        win.blit(highlight_surf, (col * square_size, row * square_size))
        
        # Draw valid move squares
        for move in valid_moves:
            row = 7 - chess.square_rank(move.to_square) if not flipped else chess.square_rank(move.to_square)
            col = chess.square_file(move.to_square) if not flipped else 7 - chess.square_file(move.to_square)
            win.blit(valid_move_surf, (col * square_size, row * square_size))