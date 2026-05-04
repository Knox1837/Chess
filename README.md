# Chess AI

A Python chess game with a trainable neural network AI. 

## Setup

**Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install chess pygame-ce numpy tqdm matplotlib pandas
```

Alternative for CPU only:
```bash
pip install torch torchvision torchaudio
pip install chess pygame-ce numpy tqdm matplotlib pandas
```

**Run:**
```bash
python main.py
```

A pre-trained model is included — select **Option 2** to play immediately.

## Menu

| Option | Description |
|--------|-------------|
| 1 | Human vs Human |
| 2 | Play vs Trained Neural Network |
| 3 | Play vs Heuristic AI |
| 4 | Train a new model |
| 5 | Play vs Stockfish |
| 6 | Exit |



## Controls

| Key | Action |
|-----|--------|
| Click / Drag | Move pieces |
| A | Toggle AI on/off |
| S | Switch AI side |
| R | Reset board |
| Z | Undo move |
| ESC | Return to menu |



## Training
**Step 1:- Get a PGN file**
Download from https://database.lichess.org/ and place it in `engine/data/raw/`.

**Step 2:- Download Stockfish for better training labels**
Download from https://stockfishchess.org/download/, rename to `stockfish.exe`, place at `engine/stockfish.exe`.

Without Stockfish, positions are labeled by game outcome (win/loss). With Stockfish, positions are labeled by centipawn evaluation which provides significantly stronger results but slower processing due to per position evaluation.

| Mode | Recommended games | Processing time |
|------|------------------|----------------|
| Win/loss labels | 50,000 | ~10 min |
| Stockfish depth 5 | 5,000 | ~15 min |
| Stockfish depth 5 | 20,000 | ~1.5 hrs |

**Step 3:- Select Option 4 from the menu**

Enter number of games and epochs (25 recommended). Model saves to `engine/models/saved/chess_ai_final.pth` when done.
