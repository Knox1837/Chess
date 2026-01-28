"""
Chess Launcher - Run everything from here
"""
import os
import sys
import subprocess
import argparse

def setup_paths():
    """Setup Python paths"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Add engine and game directories
    engine_path = os.path.join(project_root, "engine")
    game_path = os.path.join(project_root, "game")
    
    for path in [engine_path, game_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root

def run_chess_gui():
    """Run the PyGame chess GUI"""
    print("Starting Chess GUI...")
    print("Press 'A' to toggle AI opponent (if available)")
    print("Press 'R' to reset board")
    print("Press 'Ctrl+Z' to undo move")
    print("Close window to exit")
    print("-" * 50)
    
    try:
        from game.board import main
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure you're in the Chess directory and have pygame installed:")
        print("pip install pygame python-chess")
        return False
    return True

def process_pgn_data():
    """Process PGN data for training"""
    print("Processing PGN data...")
    
    pgn_file = "engine/data/raw/lichess_db_standard_rated_2013-01.pgn"
    
    if not os.path.exists(pgn_file):
        print(f"Error: PGN file not found at {pgn_file}")
        return False
    
    try:
        from engine.data.pgn_processor import LichessPGNProcessor
        processor = LichessPGNProcessor()
        processor.process_pgn_file(pgn_file, max_games=1000)
        return True
    except Exception as e:
        print(f"Error processing PGN: {e}")
        return False

def train_model():
    """Train the neural network"""
    print("Training neural network...")
    
    # Check if processed data exists
    processed_dir = "engine/data/processed"
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        print("No processed data found. Process PGN data first.")
        return False
    
    try:
        from engine.training.train_basic import main as train_main
        train_main()
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def test_ai():
    """Test the AI player"""
    print("Testing AI player...")
    
    try:
        from engine.ai_player import ChessAI
        import chess
        
        ai = ChessAI(skill_level=3)
        board = chess.Board()
        
        move_result = ai.choose_move(board)
        if move_result:
            print(f"AI suggests: {move_result['move_san']}")
            print(f"Evaluation: {move_result['eval']:.2f}")
            return True
        else:
            print("AI couldn't generate a move")
            return False
    except Exception as e:
        print(f"Error testing AI: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Chess Launcher')
    parser.add_argument('mode', choices=['gui', 'process', 'train', 'test', 'all'], 
                       nargs='?', default='gui', help='What to run')
    
    args = parser.parse_args()
    
    # Setup paths first
    setup_paths()
    
    print("=" * 60)
    print("Chess Project Launcher")
    print("=" * 60)
    
    if args.mode == 'gui':
        run_chess_gui()
    
    elif args.mode == 'process':
        process_pgn_data()
    
    elif args.mode == 'train':
        train_model()
    
    elif args.mode == 'test':
        test_ai()
    
    elif args.mode == 'all':
        print("Running complete pipeline...")
        print("\n1. Processing PGN data:")
        if process_pgn_data():
            print("\n2. Training model:")
            if train_model():
                print("\n3. Testing AI:")
                test_ai()
                print("\n4. Starting GUI with AI:")
                run_chess_gui()
        else:
            print("Failed at PGN processing step")

if __name__ == "__main__":
    main()