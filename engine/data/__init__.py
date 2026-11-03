"""
Data processing module for the chess engine.

PGNProcessor is imported lazily so that importing this package does not load pandas/tqdm.
"""

__all__ = ['LichessPGNProcessor']


def __getattr__(name):
    if name in __all__:
        from . import pgn_processor
        return getattr(pgn_processor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")