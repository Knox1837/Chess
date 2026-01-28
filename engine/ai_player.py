"""
AI Player that integrates with the chess GUI
"""
import torch
import chess
import numpy as np
import random
import sys
import os
from pathlib import Path
from engine.models.chess_net import SimpleChessNet
import sys
from pathlib import Path

# Add the engine directory to Python path
engine_dir = Path(__file__).parent
if str(engine_dir) not in sys.path:
    sys.path.append(str(engine_dir))

try:
    from models.chess_net import SimpleChessNet, PositionEvaluator
    print("✓ Successfully imported chess models")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the correct directory")
    SimpleChessNet = None
    PositionEvaluator = None


# Add path to import your game modules
sys.path.append(str(Path(__file__).parent.parent / "game"))

class ChessAI:
    """AI player that uses trained neural network"""
    def __init__(self, model_path=None, skill_level=3):
        self.skill_level = skill_level
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load neural network
        if model_path and os.path.exists(model_path):
            self.model = SimpleChessNet().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ Loaded trained AI from {model_path}")
        else:
            self.model = None
            print(f"⚠️ Using heuristic AI (skill level: {skill_level})")
    
    def board_to_tensor(self, board):
        """Convert board to tensor"""
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
        """Evaluate using neural network"""
        if self.model is None:
            return 0.0
        
        tensor = self.board_to_tensor(board)
        tensor = torch.tensor(tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.model(tensor)
        
        return value.item()
    
    def evaluate_position_heuristic(self, board):
        """Simple heuristic evaluation"""
        if board.is_checkmate():
            return -1000 if board.turn == chess.WHITE else 1000
        
        # Piece values
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_score = 0
        black_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_score += value
                else:
                    black_score += value
        
        evaluation = (white_score - black_score) / 10.0
        
        # Add randomness for lower skill levels
        if self.skill_level <= 2:
            evaluation += random.uniform(-1.0, 1.0)
        
        return evaluation
    
    def choose_move(self, board, depth=2):
        """Choose best move"""
        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # For beginner level, sometimes make random moves
        if self.skill_level == 1 and random.random() < 0.3:
            move = random.choice(legal_moves)
            return {'move': move, 'eval': 0, 'move_san': board.san(move)}
        
        for move in legal_moves:
            # Make move
            board.push(move)
            
            # Evaluate
            if self.model:
                eval_score = self.evaluate_position_nn(board)
            else:
                eval_score = self.evaluate_position_heuristic(board)
            
            # From perspective of player to move
            if board.turn == chess.BLACK:  # We just moved for white
                current_eval = eval_score
            else:  # We just moved for black
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
        
        if best_move:
            return {
                'move': best_move,
                'eval': best_value,
                'move_san': board.san(best_move)
            }
        
        return None

# Test function
def test_ai():
    """Test the AI player"""
    ai = ChessAI(skill_level=3)
    board = chess.Board()
    
    print("Testing AI with starting position:")
    print(board)
    
    # Get AI move
    result = ai.choose_move(board)
    
    if result:
        print(f"\nAI suggests: {result['move_san']}")
        print(f"Evaluation: {result['eval']:.2f}")
        print(f"Delay: {result['delay']}ms")
        
        # Make the move
        board.push(result['move'])
        print("\nPosition after AI move:")
        print(board)
    
    return ai

if __name__ == "__main__":
    test_ai()