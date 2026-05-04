"""
Chess Engine Package
"""

# Version
__version__ = "1.0.0"

# from .ai_player import ChessAI
from .ai.chess_ai import ChessAI

# Optional Export everything
__all__ = ['ChessAI']