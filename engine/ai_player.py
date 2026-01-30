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
            try:
                from models.chess_net import SimpleChessNet, PositionEvaluator
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Determine model type
                if checkpoint.get('model_class') == 'SimpleChessNet':
                    self.model = SimpleChessNet().to(self.device)
                else:
                    self.model = PositionEvaluator().to(self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print(f"AI: Loaded neural network from {model_path}")
                print(f"AI: Skill level {skill_level} (using neural network)")
            except Exception as e:
                print(f"AI: Could not load neural network: {e}")
                print("AI: Falling back to heuristic player")
                self.model = None
        else:
            print(f"AI: Skill level {skill_level} (using heuristics)")
    
    def board_to_tensor(self, board):
        """Convert board to tensor (compatible with training)"""
        # Simple implementation - you can import from pgn_processor
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
            if isinstance(self.model, SimpleChessNet):
                value, _ = self.model(tensor)
            else:
                value = self.model(tensor)
        
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
            return None
        
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
            # Add delay based on skill level (higher skill = faster)
            delay = max(0, 1000 - (self.skill_level * 200))
            
            return {
                'move': move,
                'eval': eval_score,
                'delay': delay,
                'move_san': board.san(move)
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