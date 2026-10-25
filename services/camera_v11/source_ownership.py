"""Compatibility export for the shared camera ownership primitive.

The canonical SourceOwnership implementation now lives in services.shared,
so GPU, ML and legacy camera workers share one flock contract.
"""
from services.shared.source_ownership import SourceOwnership

__all__ = ["SourceOwnership"]
