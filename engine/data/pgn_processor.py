"""
Process Lichess PGN files for ML training
"""
import chess
import chess.pgn
import numpy as np
import pandas as pd
from tqdm import tqdm # smart progress bar
import pickle
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "game"))

class LichessPGNProcessor:
    def __init__(self):
        self.processed_dir = Path(__file__).parent / "processed"
        self.processed_dir.mkdir(exist_ok=True)
    
    def board_to_tensor(self, board):
        """
        Convert chess board to 8x8x13 tensor
        Compatible with your GUI's coordinate system
        """
        # 13 channels: 6 white pieces, 6 black pieces, 1 for turn
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        
        # Piece mapping (matches your GUI's piece representation)
        piece_to_channel = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,    # White
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11   # Black
        }
        
        # Fill piece channels - using your GUI's coordinate system
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Match your game/draw_pieces() coordinate system
                # In your GUI: row = 7 - chess.square_rank(square)
                row = 7 - chess.square_rank(square)  # 0 at top, 7 at bottom
                col = chess.square_file(square)      # 0 at left, 7 at right
                channel = piece_to_channel[piece.symbol()]
                tensor[channel, row, col] = 1
        
        # Channel 12: turn (1 = white, 0 = black)
        tensor[12] = 1.0 if board.turn == chess.WHITE else 0.0
        
        return tensor
    
    def move_to_label(self, move, board):
        """Convert move to label for training"""
        # Simplified: create one-hot encoding for all possible moves
        moves = list(board.legal_moves)
        move_dict = {m.uci(): i for i, m in enumerate(moves)}
        
        label = np.zeros(len(moves), dtype=np.float32)
        if move.uci() in move_dict:
            label[move_dict[move.uci()]] = 1.0
        
        return label, move.uci()
    
    def process_pgn_file(self, pgn_path, max_games=1000, save_every=100):
        """
        Process PGN file and save training data
        
        Args:
            pgn_path: Path to PGN file
            max_games: Maximum number of games to process
            save_every: Save checkpoint every N games
        """
        print(f"Processing PGN file: {pgn_path}")
        
        all_positions = []
        all_moves = []
        all_results = []
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
                    result_value = 1.0  # White wins
                elif result_str == "0-1":
                    result_value = -1.0  # Black wins
                else:
                    result_value = 0.0   # Draw
                
                # Process each move
                board = game.board()
                for move in game.mainline_moves():
                    # Convert board to tensor
                    position_tensor = self.board_to_tensor(board)
                    
                    # Get move label
                    move_label, move_uci = self.move_to_label(move, board)
                    
                    # Determine result from perspective of player to move
                    current_result = result_value if board.turn == chess.WHITE else -result_value
                    
                    all_positions.append(position_tensor)
                    all_moves.append(move_label)
                    all_results.append(current_result)
                    
                    # Make the move
                    board.push(move)
                
                game_count += 1
                pbar.update(1)
                
                # Save checkpoint
                if game_count % save_every == 0:
                    self.save_checkpoint(all_positions, all_moves, all_results, game_count)
            
            pbar.close()
        
        # Final save
        self.save_data(all_positions, all_moves, all_results, f"full_{game_count}_games")
        print(f"\nProcessed {game_count} games, {len(all_positions)} positions")
    
    def save_checkpoint(self, positions, moves, results, game_count):
        """Save intermediate checkpoint"""
        checkpoint_dir = self.processed_dir / f"checkpoint_{game_count}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        self.save_data(positions, moves, results, f"checkpoint_{game_count}")
        print(f"Saved checkpoint after {game_count} games")
    
    def save_data(self, positions, moves, results, name):
        """Save data to numpy files"""
        BASE_DIR = Path(__file__).resolve().parent
        PROCESSED_DIR = BASE_DIR / "processed" / name
        PROCESSED_DIR.mkdir(parents= True, exist_ok=True)

        # Convert to arrays
        positions_array = np.array(positions, dtype=np.float32)
        results_array = np.array(results, dtype=np.float32)
        
        # Save positions and results
        np.save(PROCESSED_DIR / "positions.npy", positions_array)
        np.save(PROCESSED_DIR / "results.npy", results_array)
        
        # Save moves (as pickled list since they have variable length)
        with open(PROCESSED_DIR / "moves.pkl", "wb") as f:
            pickle.dump(moves, f)
        
        # Save metadata
        metadata = {
            'num_positions': len(positions_array),
            'num_games': name,
            'position_shape': positions_array.shape
        }
        with open(PROCESSED_DIR / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"Saved {len(positions)} positions to {PROCESSED_DIR}")

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Lichess PGN files')
    parser.add_argument('--pgn', type=str, required=True, help='Path to PGN file')
    parser.add_argument('--max-games', type=int, default=1000, help='Max games to process')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    processor = LichessPGNProcessor()
    processor.process_pgn_file(args.pgn, max_games=args.max_games)

if __name__ == "__main__":
    main()