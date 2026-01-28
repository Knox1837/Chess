import os
import sys
import pygame
import chess
from engine.data.pgn_processor import PGNProcessor
from engine.training.train_basic import ChessAITrainer
from game.renderer import ChessRenderer
from game.game_state import ChessGame

# Setup pygame
pygame.init()
WIN = pygame.display.set_mode((ChessRenderer.WIDTH, ChessRenderer.HEIGHT))
pygame.display.set_caption("Chess AI Trainer & Player")
CLOCK = pygame.time.Clock()

# ------------------ MAIN MENU ------------------
def show_menu():
    """Show main menu"""
    menu_items = [
        "1. Play Chess (Human vs Human)",
        "2. Play vs AI (Trained Model)",
        "3. Play vs Basic AI (Heuristic)",
        "4. Train New AI Model",
        "5. Exit"
    ]
    
    while True:
        WIN.fill((50, 50, 50))
        
        # Draw title
        title_font = pygame.font.SysFont(None, 48)
        title = title_font.render("CHESS AI SYSTEM", True, (255, 255, 255))
        WIN.blit(title, (ChessRenderer.WIDTH//2 - title.get_width()//2, 50))
        
        # Draw menu items
        font = pygame.font.SysFont(None, 36)
        for i, item in enumerate(menu_items):
            text = font.render(item, True, (200, 200, 200))
            WIN.blit(text, (100, 150 + i * 60))
        
        # Draw instructions
        inst_font = pygame.font.SysFont(None, 24)
        instructions = [
            "During game: A=toggle AI, S=switch sides, R=reset, Z=undo, ESC=menu"
        ]
        for i, inst in enumerate(instructions):
            text = inst_font.render(inst, True, (150, 150, 150))
            WIN.blit(text, (ChessRenderer.WIDTH//2 - text.get_width()//2, ChessRenderer.HEIGHT - 100 + i * 30))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 5
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 1
                elif event.key == pygame.K_2:
                    return 2
                elif event.key == pygame.K_3:
                    return 3
                elif event.key == pygame.K_4:
                    return 4
                elif event.key == pygame.K_5 or event.key == pygame.K_ESCAPE or pygame.quit():
                    return 5
        
        CLOCK.tick(60)

# ------------------ TRAINING SCREEN ------------------
def train_ai_screen():
    """Show AI training screen"""
    print("\n" + "="*60)
    print("AI TRAINING")
    print("="*60)
    
    # Ask for PGN file
    pgn_files = []
    if os.path.exists("engine/data/raw"):
        pgn_files = [f for f in os.listdir("engine/data/raw") if f.endswith('.pgn')]
    
    if not pgn_files:
        print("No PGN files found in engine/data/raw/")
        print("Please place your PGN file there and restart.")
        input("Press Enter to return to menu...")
        return
    
    print("\nAvailable PGN files:")
    for i, file in enumerate(pgn_files, 1):
        print(f"  {i}. {file}")
    
    try:
        choice = int(input("\nSelect file number: ")) - 1
        if 0 <= choice < len(pgn_files):
            pgn_path = os.path.join("engine/data/raw", pgn_files[choice])
        else:
            print("Invalid choice")
            return
    except:
        print("Invalid input")
        return
    
    # Get training parameters
    try:
        num_games = int(input("Number of games to process (100-10000): ") or "1000")
        epochs = int(input("Training epochs (5-100): ") or "20")
    except:
        num_games = 1000
        epochs = 20
    
    print(f"\nTraining with {num_games} games for {epochs} epochs...")
    print("This will take some time. Please wait...")
    
    # Process PGN
    processor = PGNProcessor()
    processor.process_pgn(pgn_path, max_games=num_games)
    
    # Train model
    trainer = ChessAITrainer()
    trainer.train(epochs=epochs, batch_size=32)
    
    print("\n✅ Training complete!")
    print("You can now play against your trained AI.")
    input("Press Enter to return to menu...")

# ------------------ GAME SCREEN ------------------
def play_game(use_trained_ai=False):
    """Play chess game"""
    # Load appropriate AI
    if use_trained_ai:
        model_path = "chess_ai_final.pth"
        if not os.path.exists(model_path):
            print("No trained model found. Using basic AI instead.")
            model_path = None
    else:
        model_path = None
    
    game = ChessGame(ai_model_path=model_path)
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # AI controls
                elif event.key == pygame.K_a:
                    game.ai_enabled = not game.ai_enabled
                    status = "enabled" if game.ai_enabled else "disabled"
                    print(f"AI {status}")
                
                elif event.key == pygame.K_s:
                    game.ai_color = chess.WHITE if game.ai_color == chess.BLACK else chess.BLACK
                    color = "White" if game.ai_color == chess.WHITE else "Black"
                    print(f"AI now plays {color}")
                
                elif event.key == pygame.K_r:
                    game.board.reset()
                    game.selected_square = None
                    game.valid_moves = []
                    print("Board reset")
                
                elif event.key == pygame.K_z:
                    if game.board.move_stack:
                        game.board.pop()
                        game.selected_square = None
                        game.valid_moves = []
                        print("Undo move")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    game.handle_click(event.pos)
        
        # Update game state (AI moves)
        game.update()
        
        # Draw everything
        WIN.fill((0, 0, 0))  # Clear screen
        game.draw(WIN)
        pygame.display.flip()
        
        CLOCK.tick(60)
    
    return True

# ------------------ MAIN FUNCTION ------------------
def main():
    """Main program loop"""
    print("="*60)
    print("CHESS AI TRAINER & PLAYER")
    print("="*60)
    print("\nInstructions:")
    print("1. Place PGN files in: engine/data/raw/")
    print("2. Train AI first (Option 4)")
    print("3. Play against trained AI (Option 2)")
    print("="*60)
    
    while True:
        choice = show_menu()
        
        if choice == 1:
            print("\nStarting Human vs Human game...")
            play_game(use_trained_ai=False)
        
        elif choice == 2:
            print("\nStarting game vs Trained AI...")
            play_game(use_trained_ai=True)
        
        elif choice == 3:
            print("\nStarting game vs Basic AI...")
            play_game(use_trained_ai=False)
        
        elif choice == 4:
            train_ai_screen()
        
        elif choice == 5:
            print("\nThanks for playing!")
            break
    
    pygame.quit()
    sys.exit()

# ------------------ QUICK START ------------------
def quick_start():
    """Quick start for immediate play"""
    print("Quick Start: Loading chess game...")
    
    # Try to load trained model
    model_path = "chess_ai_final.pth"
    if not os.path.exists(model_path):
        print("No trained AI found. Using basic AI.")
        model_path = None
    
    game = ChessGame(ai_model_path=model_path)
    game.ai_enabled = True
    
    print("\nGame Started!")
    print("Controls:")
    print("  Click - Select and move pieces")
    print("  A - Toggle AI")
    print("  S - Switch AI side")
    print("  R - Reset board")
    print("  Z - Undo move")
    print("  ESC - Quit")
    print("\n" + "="*50)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    game.ai_enabled = not game.ai_enabled
                elif event.key == pygame.K_s:
                    game.ai_color = chess.WHITE if game.ai_color == chess.BLACK else chess.BLACK
                elif event.key == pygame.K_r:
                    game.board.reset()
                elif event.key == pygame.K_z:
                    if game.board.move_stack:
                        game.board.pop()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    game.handle_click(event.pos)
        
        game.update()
        
        WIN.fill((0, 0, 0))
        game.draw(WIN)
        pygame.display.flip()
        CLOCK.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    # Check command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Chess AI System')
    parser.add_argument('--train', action='store_true', help='Train AI immediately')
    parser.add_argument('--play', action='store_true', help='Play immediately')
    parser.add_argument('--pgn', type=str, help='PGN file to process')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to process')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    
    args = parser.parse_args()
    
    if args.train:
        # Train immediately
        if args.pgn:
            pgn_path = args.pgn
        else:
            # Look for PGN in engine/data/raw
            pgn_files = []
            if os.path.exists("engine/data/raw"):
                pgn_files = [f for f in os.listdir("engine/data/raw") if f.endswith('.pgn')]
            
            if pgn_files:
                pgn_path = os.path.join("engine/data/raw", pgn_files[0])
            else:
                print("No PGN file found. Please place a .pgn file in engine/data/raw/")
                sys.exit(1)
        
        print(f"Training AI with {args.games} games from {pgn_path}...")
        processor = PGNProcessor()
        processor.process_pgn(pgn_path, max_games=args.games)
        
        trainer = ChessAITrainer()
        trainer.train(epochs=args.epochs)
        
        print("\n✅ Training complete! You can now run: python run.py --play")
    
    elif args.play:
        # Play immediately
        quick_start()
    
    else:
        # Show menu
        main()