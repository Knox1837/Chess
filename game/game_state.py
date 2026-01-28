import chess
from game.renderer import ChessRenderer
from engine.ai_player import ChessAI
import pygame

class ChessGame:
    """Main chess game logic"""
    def __init__(self, ai_model_path=None):
        self.board = chess.Board()
        self.piece_images = ChessRenderer.load_piece_images()
        self.selected_square = None
        self.valid_moves = []
        self.renderer = ChessRenderer()
        # AI setup
        self.ai = ChessAI(model_path=ai_model_path, skill_level=4 if ai_model_path else 2)
        self.ai_enabled = False
        self.ai_color = chess.BLACK
        self.ai_thinking = False
        # Highlight surfaces
        self.highlight_surf = pygame.Surface((ChessRenderer.SQUARE_SIZE, ChessRenderer.SQUARE_SIZE), pygame.SRCALPHA)
        self.highlight_surf.fill((255, 255, 0, 100))

        self.valid_move_surf = pygame.Surface((ChessRenderer.SQUARE_SIZE, ChessRenderer.SQUARE_SIZE), pygame.SRCALPHA)
        self.valid_move_surf.fill((0, 255, 0, 100))


    def get_square_from_mouse(self, pos):
        """Convert mouse position to chess square"""
        x, y = pos
        col = x // ChessRenderer.SQUARE_SIZE
        row = 7 - (y // ChessRenderer.SQUARE_SIZE)
        
        if 0 <= row < 8 and 0 <= col < 8:
            return chess.square(col, row)
        return None
    
    def handle_click(self, pos):
        """Handle mouse click for piece movement"""
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
                self.select_piece(square)
                return True
            
            # Clicked empty square - deselect
            self.selected_square = None
            self.valid_moves = []
            return False
        
        # No square selected yet - try to select a piece
        if piece:
            self.select_piece(square)
            return True
        
        return False
    
    def select_piece(self, square):
        """Select a piece and show valid moves"""
        piece = self.board.piece_at(square)
        if piece and ((self.board.turn == chess.WHITE and piece.color == chess.WHITE) or
                     (self.board.turn == chess.BLACK and piece.color == chess.BLACK)):
            self.selected_square = square
            self.valid_moves = [move for move in self.board.legal_moves 
                              if move.from_square == square]
            return True
        return False
    
    def draw(self, win):
        """Draw the game"""
        ChessRenderer.draw_board(win)
        
        # Draw highlights
        if self.selected_square is not None:
            row = 7 - chess.square_rank(self.selected_square)
            col = chess.square_file(self.selected_square)
            win.blit(self.highlight_surf, (col * ChessRenderer.SQUARE_SIZE, row * ChessRenderer.SQUARE_SIZE))
            
            for move in self.valid_moves:
                row = 7 - chess.square_rank(move.to_square)
                col = chess.square_file(move.to_square)
                win.blit(self.valid_move_surf, (col * ChessRenderer.SQUARE_SIZE, row * ChessRenderer.SQUARE_SIZE))
        
        # Draw pieces
        ChessRenderer.draw_pieces(win, self.board, self.piece_images)
        
        # Draw status
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            status = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            status = "Stalemate! Draw!"
        # elif self.board.is_check():
        #     status = "Check!"
        else:
            #status = "White to move" if self.board.turn == chess.WHITE else "Black to move"
            status = ""
        
        ChessRenderer.draw_status(win, status) #only for checkmate and stalemate
        
        # Draw controls
        # draw_controls(win, self.ai_enabled, self.ai_color, self.ai_thinking)
    
    def update(self):
        """Update game state - called each frame"""
        # AI move if enabled
        if (self.ai_enabled and not self.board.is_game_over() and 
            self.board.turn == self.ai_color and not self.ai_thinking):
            
            self.ai_thinking = True
            
            # Get AI move (in a real game you'd do this in a separate thread)
            move_result = self.ai.choose_move(self.board)
            
            if move_result:
                self.board.push(move_result['move'])
                print(f"AI plays: {move_result['move_san']} (eval: {move_result['eval']:.2f})")
            
            self.ai_thinking = False
            
            # Clear selection
            self.selected_square = None
            self.valid_moves = []
            
            return True
        
        return False
