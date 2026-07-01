"""
AI Player
"""
import torch
import chess
import chess.polyglot
import numpy as np
import random
import os
from pathlib import Path

try:
    from engine.models.chess_net import SimpleChessNet, PositionEvaluator, ChessNet
except ImportError:
    SimpleChessNet = None
    PositionEvaluator = None
# Piece-square tables (White's perspective, rank 8 to rank 1)
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK_TABLE = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
KING_TABLE = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
]
# In endgames the king becomes an active piece — reward centralization
KING_ENDGAME_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]
PIECE_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE,
}

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
        self.transposition_table = {}

        if torch.cuda.is_available():
            best = max(range(torch.cuda.device_count()),
                    key=lambda i: torch.cuda.get_device_properties(i).total_memory)
            self.device = torch.device(f'cuda:{best}')
            print(f"AI using GPU: {torch.cuda.get_device_name(best)}")
        else:
            self.device = torch.device('cpu')
            print("AI using CPU")

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
            if model_class == 'ChessNet':
                self.model = ChessNet().to(self.device)
            elif model_class == 'SimpleChessNet':
                self.model = SimpleChessNet().to(self.device)
            else:
                self.model = PositionEvaluator().to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
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

        # Only iterate occupied squares (much faster than all 64)
        for square, piece in board.piece_map().items():
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
        # Reuse pre-allocated buffer to avoid repeated GPU memory allocation per call
        if not hasattr(self, '_input_buffer'):
            self._input_buffer = torch.zeros(1, 13, 8, 8, dtype=torch.float32, device=self.device)
        self._input_buffer[0].copy_(torch.from_numpy(tensor))

        with torch.no_grad():
            model_output = self.model(self._input_buffer)
            if isinstance(model_output, tuple):
                value = model_output[0]
            else:
                value = model_output

        return value.item()
    
    def _is_endgame(self, board):
        """True when both sides have no queens, or one side has queen + no other pieces."""
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        minor_and_rooks = sum(
            len(board.pieces(pt, c))
            for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK)
            for c in (chess.WHITE, chess.BLACK)
        )
        return queens == 0 or (queens <= 1 and minor_and_rooks <= 1)

    def evaluate_position_heuristic(self, board):
        """Heuristic evaluation from White's perspective."""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }

        endgame = self._is_endgame(board)
        score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
            value = piece_values[piece.piece_type]
            # Use endgame king table when appropriate
            if piece.piece_type == chess.KING and endgame:
                table = KING_ENDGAME_TABLE
            else:
                table = PIECE_TABLES.get(piece.piece_type)
            if table:
                idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
                table_idx = (7 - chess.square_rank(idx)) * 8 + chess.square_file(idx)
                value += table[table_idx] * 0.1
            score += value if piece.color == chess.WHITE else -value

        # Mop-up bonus: in endgame, push losing king to corner and close in
        if endgame:
            white_material = sum(
                piece_values[pt] * len(board.pieces(pt, chess.WHITE))
                for pt in piece_values if pt != chess.KING
            )
            black_material = sum(
                piece_values[pt] * len(board.pieces(pt, chess.BLACK))
                for pt in piece_values if pt != chess.KING
            )
            if white_material > black_material + 200:
                # White is winning: push black king to corner, bring white king close
                bk_sq = board.king(chess.BLACK)
                wk_sq = board.king(chess.WHITE)
                if bk_sq is not None and wk_sq is not None:
                    bk_r, bk_f = chess.square_rank(bk_sq), chess.square_file(bk_sq)
                    wk_r, wk_f = chess.square_rank(wk_sq), chess.square_file(wk_sq)
                    # Reward cornering the enemy king (distance from centre)
                    corner_bonus = (3 - min(bk_r, 7 - bk_r)) + (3 - min(bk_f, 7 - bk_f))
                    # Reward proximity of white king to black king
                    proximity_bonus = 14 - (abs(wk_r - bk_r) + abs(wk_f - bk_f))
                    score += (corner_bonus * 10 + proximity_bonus * 5) / 100.0
            elif black_material > white_material + 200:
                # Black is winning: mirror logic
                wk_sq = board.king(chess.WHITE)
                bk_sq = board.king(chess.BLACK)
                if wk_sq is not None and bk_sq is not None:
                    wk_r, wk_f = chess.square_rank(wk_sq), chess.square_file(wk_sq)
                    bk_r, bk_f = chess.square_rank(bk_sq), chess.square_file(bk_sq)
                    corner_bonus = (3 - min(wk_r, 7 - wk_r)) + (3 - min(wk_f, 7 - wk_f))
                    proximity_bonus = 14 - (abs(bk_r - wk_r) + abs(bk_f - wk_f))
                    score -= (corner_bonus * 10 + proximity_bonus * 5) / 100.0

        # Mobility bonus
        if board.turn == chess.WHITE:
            white_mobility = len(list(board.legal_moves))
            board.push(chess.Move.null())
            black_mobility = len(list(board.legal_moves))
            board.pop()
        else:
            black_mobility = len(list(board.legal_moves))
            board.push(chess.Move.null())
            white_mobility = len(list(board.legal_moves))
            board.pop()
        score += (white_mobility - black_mobility) * 0.1

        # Doubled pawn penalty
        for color in [chess.WHITE, chess.BLACK]:
            file_counts = [0] * 8
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    file_counts[chess.square_file(square)] += 1
            penalty = sum((c - 1) * 20 for c in file_counts if c > 1)
            score -= penalty if color == chess.WHITE else -penalty

        # Normalise to ~-10..+10 (compatible with NN output range)
        score = score / 100.0

        # Randomness for lower skill levels
        if self.skill_level <= 2:
            score += random.uniform(-0.5, 0.5)

        return score
    
    def evaluate_position(self, board):
        """Always returns score from White's perspective (positive = White is better)."""
        if self.model and self.skill_level >= 4:
            return self.evaluate_position_nn(board)
        else:
            return self.evaluate_position_heuristic(board)
    
    def order_moves(self, board):
        """Return legal moves sorted: captures > checks > promotions > quiet."""
        victim_val = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        def priority(move):
            score = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10 * victim_val.get(victim.piece_type, 0) \
                            - victim_val.get(attacker.piece_type, 0)
                else:
                    score += 10
            if board.gives_check(move):
                score += 5
            if move.promotion:
                score += 8
            return score
        return sorted(board.legal_moves, key=priority, reverse=True)
    
    def quiescence(self, board, alpha, beta, depth=0):
        """Search captures and checks until position is quiet."""
        raw = self.evaluate_position(board)
        stand_pat = raw if board.turn == chess.WHITE else -raw

        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        for move in board.legal_moves:
            # In endgames also search checks; always limit check depth to avoid explosion
            is_check = board.gives_check(move)
            if not board.is_capture(move) and not (is_check and depth < 2):
                continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, depth + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    def get_best_move(self, board, depth=1, alpha=-float('inf'), beta=float('inf')):
        actual_depth = min(depth, self.skill_level)

        # Transposition table lookup
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table:
            cached_depth, cached_move, cached_value = self.transposition_table[board_hash]
            if cached_depth >= actual_depth:
                return cached_move, cached_value

        if board.is_game_over():
            raw = self.evaluate_position(board)
            return None, raw if board.turn == chess.WHITE else -raw

        if actual_depth == 0:
            return None, self.quiescence(board, alpha, beta)

        legal_moves = self.order_moves(board)

        if not legal_moves:
            return None, 0

        if self.skill_level == 1 and random.random() < 0.3:
            return random.choice(list(board.legal_moves)), 0

        best_move = None
        best_value = -float('inf')

        for move in legal_moves:
            board.push(move)
            _, move_value = self.get_best_move(board, actual_depth - 1, -beta, -alpha)
            eval_score = -move_value
            board.pop()

            if eval_score > best_value:
                best_value = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break

        # Store in transposition table
        self.transposition_table[board_hash] = (actual_depth, best_move, best_value)
        if len(self.transposition_table) > 500_000:
            self.transposition_table.clear()

        return best_move, best_value
    
    def choose_move(self, board):
        # Don't clear transposition table every move — reuse cached positions across turns
        # The overflow check inside get_best_move handles memory management

        piece_count = len(board.piece_map())
        if piece_count <= 8:
            # Deep endgame: search much deeper, heuristics are reliable here
            depth = max(5, self.skill_level + 2)
        elif piece_count <= 14:
            # Late middlegame / early endgame
            depth = max(3, self.skill_level)
        else:
            # Middlegame: original behaviour (NN eval is most useful here)
            depth = 2 if self.model else max(2, self.skill_level)

        move, eval_score = self.get_best_move(board, depth=depth)

        if move:
            # Convert from [-1, +1] to chess.com-style decimal (e.g. +3.2, -1.5)
            # Training used score/500, so *500 gives centipawns, /100 gives pawns
            if abs(eval_score) >= 9000:
                display_eval = f"#{int(abs(eval_score) / 1000)}"
            else:
                display_eval = f"{eval_score * 5:+.1f}"
            return {
                'move': move,
                'eval': eval_score,
                'eval_display': display_eval,
                'move_san': board.san(move)
            }
        return None