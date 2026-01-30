"""
Menu functions
"""
import pygame
from utils.config import *

def show_menu(win, clock):
    """Show main menu"""
    menu_items = [
        "1. Play Chess (Human vs Human)",
        "2. Play vs AI (Trained Model)",
        "3. Play vs Basic AI (Heuristic)",
        "4. Train New AI Model",
        "5. Exit"
    ]
    
    while True:
        win.fill((50, 50, 50))
        
        # Draw title
        title_font = pygame.font.SysFont(None, 48)
        title = title_font.render("CHESS AI SYSTEM", True, (255, 255, 255))
        win.blit(title, (WIDTH//2 - title.get_width()//2, 50))
        
        # Draw menu items
        font = pygame.font.SysFont(None, 36)
        for i, item in enumerate(menu_items):
            text = font.render(item, True, (200, 200, 200))
            win.blit(text, (100, 150 + i * 60))
        
        # Draw instructions
        inst_font = pygame.font.SysFont(None, 24)
        instructions = [
            "During game: A=toggle AI, S=switch sides, R=reset, Z=undo, ESC=menu"
        ]
        for i, inst in enumerate(instructions):
            text = inst_font.render(inst, True, (150, 150, 150))
            win.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT - 100 + i * 30))
        
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
                elif event.key == pygame.K_5 or event.key == pygame.K_ESCAPE:
                    return 5
        
        clock.tick(60)

def train_ai_screen():
    """Show AI training screen"""
    print("\n" + "="*60)
    print("AI TRAINING")
    print("="*60)
    
    try:
        from engine.data.pgn_processor import LichessPGNProcessor as PGNProcessor
        from engine.ai.chess_trainer import ChessTrainer
    except ImportError as e:
        print(f"Error: Training modules not available: {e}")
        input("Press Enter to return to menu...")
        return
    
    # Ask for PGN file
    import os
    from pathlib import Path
    
    pgn_files = []
    raw_dir = Path("engine/data/raw")
    if raw_dir.exists():
        pgn_files = [f for f in os.listdir(raw_dir) if f.endswith('.pgn')]
    
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
    processor.process_pgn_file(pgn_path, max_games=num_games)
    
    # Train model
    trainer = ChessTrainer(model_type="simple")
    trainer.train(num_epochs=epochs, batch_size=32)
    
    print("\n✅ Training complete!")
    print("You can now play against your trained AI.")
    input("Press Enter to return to menu...")