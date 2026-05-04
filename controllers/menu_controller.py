"""
Menu Controller - handles menu navigation
"""
import pygame
import sys
import os
from pathlib import Path

class MenuController:
    """Controls menu navigation and training"""
    
    def __init__(self, win, clock):
        from ui.menu import show_menu, train_ai_screen
        from controllers.game_controller import GameController
        
        self.win = win
        self.clock = clock
        self.show_menu = show_menu
        self.train_ai_screen = train_ai_screen
        self.GameController = GameController
    
    def run(self):
        """Main menu loop"""
        print("="*60)
        print("CHESS AI TRAINER & PLAYER")
        print("="*60)
        print("\nInstructions:")
        print("1. Place PGN files in: engine/data/raw/")
        print("2. Train AI first (Option 4)")
        print("3. Play against trained AI (Option 2)")
        print("4. Play vs Stockfish (Option 5)")
        print("5. Exit (Option 6 or ESC)")
        print("="*60)
        
        while True:
            choice = self.show_menu(self.win, self.clock)
            
            if choice == 1:
                print("\nStarting Human vs Human game...")
                self.start_game(use_trained_ai=False)
            
            elif choice == 2:
                print("\nStarting game vs Trained AI...")
                self.start_game(use_trained_ai=True)
            
            elif choice == 3:
                print("\nStarting game vs Basic AI...")
                game = self.GameController(self.win, self.clock, use_trained_ai=False)
                game.game.ai_enabled = True
                game.run()
            
            elif choice == 4:
                self.train_ai_screen()
            
            elif choice ==5:
                self.start_stockfish_game()

            elif choice == 6:
                self.estimate_elo()
            
            elif choice == 7:
                print("\n Thanks for playing\n")
                break
        
        pygame.quit()
        sys.exit()
    
    def start_game(self, use_trained_ai=False):
        """Start a new game"""
        game = self.GameController(self.win, self.clock, use_trained_ai)
        game.run()
    
    def quick_start(self):
        """Quick start for immediate play"""
        print("Quick Start: Loading chess game...")
        
        # Try to load trained model
        model_path = os.path.join("engine", "models", "saved", "chess_ai_final.pth")
        if not os.path.exists(model_path):
            print("No trained AI found. Using basic AI.")
            model_path = None
        
        game = self.GameController(self.win, self.clock, model_path is not None)
        game.game.ai_enabled = True
        
        print("\nGame Started!")
        print("Controls:")
        print("  Click/Drag - Select and move pieces")
        print("  A - Toggle AI")
        print("  S - Switch AI side")
        print("  R - Reset board")
        print("  Z - Undo move")
        print("  ESC - Quit")
        print("\n" + "="*50)
        
        game.run()
    
    def start_stockfish_game(self):
        """Play directly against Stockfish engine"""
        import chess.engine
        from pathlib import Path

        stockfish_path = Path("engine/stockfish.exe")
        if not stockfish_path.exists():
            print("Stockfish not found at engine/stockfish.exe")
            print("Please download from https://stockfishchess.org/download/")
            input("Press Enter to return to menu...")
            return

        game = self.GameController(self.win, self.clock, use_trained_ai=False)
        game.game.ai_enabled = True

        # Patch the AI's choose_move to use Stockfish directly
        engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))

        original_choose_move = game.game.ai.choose_move

        def stockfish_move(board):
            try:
                result = engine.play(board, chess.engine.Limit(time=1.0))
                return {
                    'move': result.move,
                    'eval': 0,
                    'move_san': board.san(result.move)
                }
            except Exception as e:
                print(f"Stockfish error: {e}")
                return original_choose_move(board)

        game.game.ai.choose_move = stockfish_move

        print("Playing vs Stockfish (1 second per move)")
        print("Controls: A=toggle AI, S=switch sides, R=reset, Z=undo, ESC=menu")

        game.run()
        engine.quit()

    def estimate_elo(self):
        """Estimate AI Elo by playing against Stockfish at different levels"""
        import chess.engine
        from pathlib import Path

        print("\n" + "="*60)
        print("AI ELO ESTIMATOR")
        print("="*60)

        stockfish_path = Path("engine/stockfish.exe")
        if not stockfish_path.exists():
            print("Stockfish not found at engine/stockfish.exe")
            print("Please download from https://stockfishchess.org/download/")
            input("Press Enter to return to menu...")
            return

        model_path = Path("engine/models/saved/chess_ai_final.pth")
        if not model_path.exists():
            print("No trained model found. Please train a model first (Option 4).")
            input("Press Enter to return to menu...")
            return

        try:
            num_games = int(input("Games per Elo level (5-50, recommend 10): ") or "10")
            num_games = max(5, min(50, num_games))
        except ValueError:
            num_games = 20

        from engine.ai.chess_ai import ChessAI
        import chess

        ai = ChessAI(model_path=str(model_path), skill_level=4)
        engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))

        test_elos = [1320, 1400, 1500, 1600, 1800] #1320 is the min allowed by stockfish used
        results = {}

        print(f"\nPlaying {num_games} games at each Elo level...")
        print("This will take a while. Press Ctrl+C to stop early.\n")

        try:
            for target_elo in test_elos:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo})
                wins = draws = losses = 0

                for game_num in range(num_games):
                    board = chess.Board()
                    ai_is_white = (game_num % 2 == 0)

                    while not board.is_game_over():
                        if (board.turn == chess.WHITE) == ai_is_white:
                            move_result = ai.choose_move(board)
                            if move_result:
                                board.push(move_result['move'])
                            else:
                                break
                        else:
                            sf_result = engine.play(board, chess.engine.Limit(time=0.01))
                            board.push(sf_result.move)

                    outcome = board.outcome()
                    if outcome:
                        if outcome.winner is None:
                            draws += 1
                        elif (outcome.winner == chess.WHITE) == ai_is_white:
                            wins += 1
                        else:
                            losses += 1

                total = wins + draws + losses
                score = (wins + 0.5 * draws) / total if total > 0 else 0
                results[target_elo] = score

                print(f"vs Stockfish {target_elo}: "
                    f"{wins}W {draws}D {losses}L  "
                    f"score={score:.2f}  "
                    f"{'→ Stronger ↑' if score > 0.6 else '→ Roughly equal ≈' if score > 0.4 else '→ Weaker ↓'}")

                # Stop early if clearly outmatched
                if score < 0.15:
                    print(f"  Stopping:- AI is too weak for higher levels")
                    break
                if score > 0.85:
                    print(f"  Clearly stronger — skipping remaining lower levels")
                    # continue to next level faster

        except KeyboardInterrupt:
            print("\nStopped early.")

        engine.quit()

        # Estimate Elo from results
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)

        estimated_elo = None
        for elo, score in sorted(results.items()):
            if 0.4 <= score <= 0.6:
                estimated_elo = elo
                break
            elif score < 0.4 and estimated_elo is None:
                # Interpolate between this and previous level
                elos = sorted(results.keys())
                idx = elos.index(elo)
                if idx > 0:
                    prev_elo = elos[idx - 1]
                    prev_score = results[prev_elo]
                    # Linear interpolation
                    t = (0.5 - score) / (prev_score - score) if prev_score != score else 0.5
                    estimated_elo = int(elo + t * (prev_elo - elo))
                break

        if estimated_elo:
            print(f"Estimated Elo: ~{estimated_elo}")
        else:
            scores = list(results.values())
            elos = list(results.keys())
            if all(s > 0.6 for s in scores):
                print(f"Estimated Elo: >{max(elos)} (stronger than all tested levels)")
            elif all(s < 0.4 for s in scores):
                print(f"Estimated Elo: <{min(elos)} (weaker than all tested levels)")

        input("\nPress Enter to return to menu...")