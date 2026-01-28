"""
Process Lichess PGN files for ML training
"""
import chess
import chess.pgn
import numpy as np
from tqdm import tqdm # smart progress bar
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "game"))

class PGNProcessor:
    """Process PGN files for training"""
    def __init__(self):
        self.processed_dir = Path("processed_data")
        self.processed_dir.mkdir(exist_ok=True)
    
    def board_to_tensor(self, board):
        """Convert chess board to 8x8x13 tensor"""
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
        
        # Channel 12: turn (1 = white, 0 = black)
        tensor[12] = 1.0 if board.turn == chess.WHITE else 0.0
        
        return tensor
    
    def process_pgn(self, pgn_path, max_games=100):
        """Process PGN file into training data"""
        print(f"Processing {pgn_path}...")
        
        positions = []
        results = []
        game_count = 0
        
        with open(pgn_path, 'r') as f:
            pbar = tqdm(total=max_games, desc="Processing games")
            
            while game_count < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Get game result
                result_str = game.headers.get("Result", "")
                if result_str == "1-0":
                    game_result = 1.0
                elif result_str == "0-1":
                    game_result = -1.0
                else:
                    game_result = 0.0
                
                # Process each position
                board = game.board()
                for move in game.mainline_moves():
                    # Save position before move
                    position_tensor = self.board_to_tensor(board)
                    
                    # Determine result from perspective of player to move
                    current_result = game_result if board.turn == chess.WHITE else -game_result
                    
                    positions.append(position_tensor)
                    results.append(current_result)
                    
                    board.push(move)
                
                game_count += 1
                pbar.update(1)
            
            pbar.close()
        
        # Save data
        positions_array = np.array(positions, dtype=np.float32)
        results_array = np.array(results, dtype=np.float32)
        
        np.save(self.processed_dir / "positions.npy", positions_array)
        np.save(self.processed_dir / "results.npy", results_array)
        
        print(f"Saved {len(positions)} positions from {game_count} games")
        return positions_array, results_array
