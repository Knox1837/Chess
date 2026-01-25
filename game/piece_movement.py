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
    
    def handle_click(self, pos):
        """Handle click for piece movement (simple two-click method)"""
        square = self.get_square_from_mouse(pos)
        
        if square is None:
            return False
        
        piece = self.board.piece_at(square)
        
        # CASE 1: Clicking on an empty square when a piece is selected
        if self.selected_square is not None and square != self.selected_square:
            # Try to move to this square
            for move in self.valid_moves:
                if move.to_square == square:
                    self.board.push(move)
                    self.last_move = move
                    self.selected_square = None
                    self.valid_moves = []
                    return True
            
            # If clicked empty square but not a valid move, select the piece at that square if any
            if piece:
                # Check if it's current player's piece
                if (self.board.turn == chess.WHITE and piece.color == chess.WHITE) or \
                   (self.board.turn == chess.BLACK and piece.color == chess.BLACK):
                    self.selected_square = square
                    self.valid_moves = [move for move in self.board.legal_moves 
                                      if move.from_square == square]
                    return True
            
            # Clicked empty square that's not a valid move - deselect
            self.selected_square = None
            self.valid_moves = []
            return False
        
        # CASE 2: Clicking on a piece
        if piece:
            # If clicking on a piece when another piece is selected
            if self.selected_square is not None:
                # If clicking same piece again, deselect it
                if square == self.selected_square:
                    self.selected_square = None
                    self.valid_moves = []
                    return True
                
                # If clicking different piece, check if it's a valid capture
                for move in self.valid_moves:
                    if move.to_square == square:
                        self.board.push(move)
                        self.last_move = move
                        self.selected_square = None
                        self.valid_moves = []
                        return True
            
            # Select this piece (if it's current player's turn)
            if (self.board.turn == chess.WHITE and piece.color == chess.WHITE) or \
               (self.board.turn == chess.BLACK and piece.color == chess.BLACK):
                self.selected_square = square
                self.valid_moves = [move for move in self.board.legal_moves 
                                  if move.from_square == square]
                return True
        
        # CASE 3: Clicking empty square with nothing selected - do nothing
        return False

    def handle_mouse_down(self, pos):
        """Handle mouse button down events"""
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
                    
                    self.dragged_piece = piece# For dragging visualization
                    self.drag_pos = pos
                    return True  # Piece selected
                else:
                    # If clicking opponent's piece when not selected, deselect
                    self.selected_square = None
                    self.valid_moves = []
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
        """Handle mouse button up events (for drag and drop)"""
        if self.selected_square is not None and self.drag_pos is not None:
            square = self.get_square_from_mouse(pos)
            
            if square is not None and square != self.selected_square:
                move = None
                for legal_move in self.valid_moves: # Find the matching move
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
    
    def handle_mouse_motion(self, pos):
        """Handle mouse motion for dragging"""
        if self.selected_square is not None and self.dragged_piece is not None:
            self.drag_pos = pos
    
    def handle_keydown(self, event): # for undo and reset
        """Handle keyboard events"""
        if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
            if self.board.move_stack:# Ctrl+Z to undo move
                self.board.pop()
                self.reset_selection()
                return True
        elif event.key == pygame.K_r:
            # R to reset the board
            self.board.reset()
            self.reset_selection()
            return True
        return False
    
    def reset_selection(self):
        """Reset all selection states"""
        self.selected_square = None
        self.valid_moves = []
        self.dragged_piece = None
        self.drag_pos = None
    
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
        if self.dragged_piece is not None and self.drag_pos is not None:
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
        if self.board.is_checkmate(): # from chess library
            return "Checkmate!"
        elif self.board.is_stalemate():
            return "Stalemate!"
        elif self.board.is_check():
            return "Check!"
        else:
            #return "White to move" if self.board.turn == chess.WHITE else "Black to move"
            return
    
    def reset_game(self):
        """Reset the entire game"""
        self.board.reset()
        self.reset_selection()