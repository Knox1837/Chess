import pygame
import chess

class ChessMovement:
    def __init__(self, board, square_size, width, height):
        self.board = board
        self.square_size = square_size
        self.width = width
        self.height = height
        self.selected_square = None
        self.valid_moves = []
        self.dragged_piece = None
        self.drag_pos = None
        self.is_dragging = False
        # Colors for highlights
        self.highlight_color = (255, 255, 0, 100)  # Yellow with transparency
        self.valid_move_color = (0, 255, 0, 100)   # Green with transparency
        
        # Create transparent surfaces for highlights
        self.highlight_surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA) # Enable per-pixel alpha
        self.valid_move_surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
    
    def get_square_from_mouse(self, pos):
        """Convert mouse position to chess square"""
        x, y = pos
        col = x // self.square_size
        row = 7 - (y // self.square_size)  # Invert because pygame y=0 is top
        
        if 0 <= row < 8 and 0 <= col < 8:
            return chess.square(col, row)
        return None
    
    def make_move(self, to_square):
        """Try to make a move from selected square to destination square"""
        if self.selected_square is None:
            return False
            
        for move in self.valid_moves:
            if move.to_square == to_square:
                self.board.push(move)
                self.reset_selection()
                return True
        return False

    def select_piece(self, square):
        """Select a piece and get its valid moves"""
        piece = self.board.piece_at(square)
        if not piece:
            return False
            
        # Check if it's the current player's piece
        if (self.board.turn == chess.WHITE and piece.color == chess.WHITE) or \
           (self.board.turn == chess.BLACK and piece.color == chess.BLACK):
            
            self.selected_square = square
            self.valid_moves = [move for move in self.board.legal_moves 
                              if move.from_square == square]
            self.dragged_piece = piece
            return True
        return False

    def handle_click(self, pos):
        """Handle click for piece movement (two-click method)"""
        square = self.get_square_from_mouse(pos)
        
        if square is None:
            return False
        
        piece = self.board.piece_at(square)
        
        # If we already have a piece selected
        if self.selected_square is not None:
            # Clicking the same piece again - deselect it
            if square == self.selected_square:
                self.reset_selection()
                return True
            
            # Clicking a different square - try to move
            if self.make_move(square):
                return True
            
            # If move failed, maybe clicked a different piece
            if piece:
                self.select_piece(square)
                return True
            
            # Clicked empty square that's not a valid move
            self.reset_selection()
            return False
        
        # No piece selected yet - try to select one
        if piece:
            return self.select_piece(square)
        
        return False
    
    def handle_mouse_down(self, pos):
        """Handle mouse button down events - start drag"""
        square = self.get_square_from_mouse(pos)
        
        if square is not None:
            piece = self.board.piece_at(square)
            
            if piece: # If a square with a piece is clicked
                # If it's the current player's piece
                if (self.board.turn == chess.WHITE and piece.color == chess.WHITE) or \
                   (self.board.turn == chess.BLACK and piece.color == chess.BLACK):
                    
                    self.selected_square = square
                    self.valid_moves = [move for move in self.board.legal_moves 
                                      if move.from_square == square]
                    
                    self.dragged_piece = piece  # For dragging visualization
                    self.drag_pos = pos
                    self.is_dragging = True  # Set dragging flag
                    return True  # Piece selected
                else:
                    # If clicking opponent's piece when not selected, deselect
                    self.selected_square = None
                    self.valid_moves = []
                    self.is_dragging = False
                    return False
            else:
                # If clicking empty square when a piece is selected, try to move
                if self.selected_square is not None:
                    move = None
                    # Find the matching move
                    for legal_move in self.valid_moves:
                        if legal_move.to_square == square:
                            move = legal_move
                            break
                    
                    if move:
                        self.board.push(move)
                        self.reset_selection()
                        return True  # Move made
                
                self.reset_selection()
                return False
        
        return False
    
    def handle_mouse_up(self, pos):
        """Complete drag operation"""
        if self.selected_square is not None and self.is_dragging:
            square = self.get_square_from_mouse(pos)
            
            if square is not None:
                # Try to make the move
                if self.make_move(square):
                    self.is_dragging = False
                    return True
            
            # Cancel drag if invalid
            self.reset_selection()
            self.is_dragging = False
            return False
        
        return False
    
    def handle_mouse_motion(self, pos):
        """Update drag position for drag and drop"""
        if self.is_dragging and self.dragged_piece is not None:
            self.drag_pos = pos
    
    def handle_keydown(self, event):
        """Handle keyboard events"""
        if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
            if self.board.move_stack:
                self.board.pop()
                self.reset_selection()
                self.is_dragging = False
                return True
        elif event.key == pygame.K_r:
            self.board.reset()
            self.reset_selection()
            self.is_dragging = False
            return True
        elif event.key == pygame.K_ESCAPE:
            # Cancel selection/drag
            self.reset_selection()
            self.is_dragging = False
            return True
        return False
    
    def reset_selection(self):
        """Reset all selection states"""
        self.selected_square = None
        self.valid_moves = []
        self.dragged_piece = None
        self.drag_pos = None
        self.is_dragging = False

    def draw_highlights(self, win):
        """Draw highlights for selected square and valid moves"""
        if self.selected_square is not None:
            # Draw highlight on selected square
            row = 7 - chess.square_rank(self.selected_square)
            col = chess.square_file(self.selected_square)
            self.highlight_surface.fill(self.highlight_color)
            win.blit(self.highlight_surface, (col * self.square_size, row * self.square_size))
        
        if self.valid_moves:
            # Draw highlights for valid move squares
            for move in self.valid_moves:
                row = 7 - chess.square_rank(move.to_square)
                col = chess.square_file(move.to_square)
                self.valid_move_surface.fill(self.valid_move_color)
                win.blit(self.valid_move_surface, (col * self.square_size, row * self.square_size))

    def draw_dragged_piece(self, win, piece_images):
        """Draw the piece being dragged on top of everything"""
        if self.is_dragging and self.dragged_piece is not None and self.drag_pos is not None:
            symbol = self.dragged_piece.symbol()
            color = 'w' if self.dragged_piece.color == chess.WHITE else 'b'
            if symbol.lower() == "n":
                key = f"N{color}"
            else:
                key = f"{symbol.upper() if color == 'w' else symbol.lower()}{color}"
            
            if key in piece_images:
                # Center the piece on the cursor
                piece_rect = piece_images[key].get_rect(center=self.drag_pos)
                win.blit(piece_images[key], piece_rect)

    def get_game_status(self):
        """Get current game status as text"""
        if self.board.is_checkmate():
            return "Checkmate!"
        elif self.board.is_stalemate():
            return "Stalemate!"
        elif self.board.is_check():
            return "Check!"
        else:
           # return "White to move" if self.board.turn == chess.WHITE else "Black to move"
           return

    
    def reset_game(self):
        """Reset the entire game"""
        self.board.reset()
        self.reset_selection()