
# Version
__version__ = "1.0.0"

# Export commonly used classes for easier imports
from .renderer import ChessRenderer
from .game_state import ChessGame

# Optional: Export everything
__all__ = ['ChessRenderer', 'ChessAI']