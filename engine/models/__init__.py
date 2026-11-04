"""
Neural network models for the chess engine.

The names below are imported lazily: for better performance, the actual model classes are only loaded when accessed.
"""

__all__ = ['SimpleChessNet', 'PositionEvaluator']


def __getattr__(name):
    if name in __all__:
        from . import chess_net
        return getattr(chess_net, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")