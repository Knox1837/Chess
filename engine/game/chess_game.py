"""
Main chess game logic
"""
import chess
import pygame
import threading
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
        self.ai_move_result = None   # Result posted here by background thread
        
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
            self.board.turn == self.ai_color):

            # If a background search just finished, apply the result
            if self.ai_move_result is not None:
                move_result = self.ai_move_result
                self.ai_move_result = None
                self.ai_thinking = False
                if move_result:
                    self.board.push(move_result['move'])
                    eval_str = move_result.get('eval_display') or f"{move_result['eval']:.2f}"
                    print(f"AI plays: {move_result['move_san']} (eval: {eval_str})")
                self.movement.selected_square = None
                self.movement.valid_moves = []
                return True

            # Start a background thread for the search if not already thinking
            if not self.ai_thinking:
                self.ai_thinking = True
                board_copy = self.board.copy()  # Thread gets its own copy — never touch self.board from thread

                def _think():
                    result = self.ai.choose_move(board_copy)
                    self.ai_move_result = result  # Post result; main thread applies it next frame

                threading.Thread(target=_think, daemon=True).start()
        
        return False