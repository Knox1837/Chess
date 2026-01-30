"""
AI Player
"""
import torch
import chess
import numpy as np
import random
import os
from pathlib import Path

try:
    from engine.models.chess_net import SimpleChessNet, PositionEvaluator
except ImportError:
    SimpleChessNet = None
    PositionEvaluator = None

class ChessAI:
    """
    AI player that can play against human in your GUI
    Can use neural network or simple heuristics
    """
    def __init__(self, model_path=None, skill_level=1):
        """
        Args:
            model_path: Path to trained PyTorch model
            skill_level: 1 (beginner) to 5 (expert)
        """
        self.skill_level = skill_level
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Load neural network if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"AI: Skill level {skill_level} (using heuristics)")
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            if SimpleChessNet is None:
                print("AI: Neural network models not available")
                return
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine model type
            model_class = checkpoint.get('model_class', 'SimpleChessNet')
            if model_class == 'SimpleChessNet':
                self.model = SimpleChessNet().to(self.device)
            else:
                self.model = PositionEvaluator().to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"AI: Loaded neural network from {model_path}")
            print(f"AI: Skill level {self.skill_level} (using neural network)")
        except Exception as e:
            print(f"AI: Could not load neural network: {e}")
            print("AI: Falling back to heuristic player")
            self.model = None
    
    def board_to_tensor(self, board):
        """Convert board to tensor (compatible with training)"""
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        
        piece_to_channel = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                channel = piece_to_channel[piece.symbol()]
                tensor[channel, row, col] = 1
        
        tensor[12] = 1.0 if board.turn == chess.WHITE else 0.0
        return tensor
    
    def evaluate_position_nn(self, board):
        """Evaluate position using neural network"""
        if self.model is None:
            return 0.0
        
        tensor = self.board_to_tensor(board)
        tensor = torch.tensor(tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(tensor)
            
            # Handle both tuple output and single output
            if isinstance(model_output, tuple):
                value = model_output[0]  # Take value from tuple
            else:
                value = model_output
        
        return value.item()
    
    def evaluate_position_heuristic(self, board):
        """Simple heuristic evaluation"""
        if board.is_checkmate():
            return -1000 if board.turn == chess.WHITE else 1000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Piece values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Count material
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Add some positional bonuses based on skill level
        position_bonus = 0
        if self.skill_level >= 3:
            # Encourage center control
            center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
            for square in center_squares:
                piece = board.piece_at(square)
                if piece:
                    if piece.color == chess.WHITE:
                        position_bonus += 0.1
                    else:
                        position_bonus -= 0.1
        
        # Calculate evaluation from white's perspective
        evaluation = (white_material - black_material) / 10.0 + position_bonus
        
        # Add randomness for lower skill levels
        if self.skill_level <= 2:
            evaluation += random.uniform(-1.0, 1.0)
        
        return evaluation
    
    def evaluate_position(self, board):
        """Evaluate position using appropriate method"""
        if self.model and self.skill_level >= 4:
            return self.evaluate_position_nn(board)
        else:
            return self.evaluate_position_heuristic(board)
    
    def get_best_move(self, board, depth=1):
        """
        Get best move using simple search
        Higher skill levels = deeper search
        """
        actual_depth = min(depth, self.skill_level)
        
        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None, 0
        
        # For beginner level, sometimes make random moves
        if self.skill_level == 1 and random.random() < 0.3:
            return random.choice(legal_moves), 0
        
        # For each legal move
        for move in legal_moves:
            # Make the move
            board.push(move)
            
            # Evaluate position
            if actual_depth > 1 and board.legal_moves:
                # Recursive search
                _, move_value = self.get_best_move(board, actual_depth - 1)
                eval_score = -move_value  # Negamax: opponent's best is our worst
            else:
                eval_score = self.evaluate_position(board)
            
            # For black, we want lower scores
            if board.turn == chess.BLACK:  # If we just made a white move
                current_eval = eval_score
            else:  # If we just made a black move
                current_eval = -eval_score
            
            # Undo move
            board.pop()
            
            # Update best move
            if board.turn == chess.WHITE:
                if current_eval > best_value:
                    best_value = current_eval
                    best_move = move
            else:
                if current_eval < best_value:
                    best_value = current_eval
                    best_move = move
        
        return best_move, best_value
    
    def choose_move(self, board):
        """Choose a move to play"""
        move, eval_score = self.get_best_move(board, depth=2)
        
        if move:
            return {
                'move': move,
                'eval': eval_score,
                'move_san': board.san(move)
            }
        
        return None