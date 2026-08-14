"""Core-v1 surveillance runtime bootstrap.

The detector hot path stays untouched. ReID is upgraded through an explicit
package-level runtime alias so existing imports of ``reid_service.ReIDCoordinator``
receive the hardened implementation without duplicating the service module.
"""

from . import reid_service as _reid_service
from .reid_hardening import HardenedReIDCoordinator

_reid_service.ReIDCoordinator = HardenedReIDCoordinator

__all__ = ["HardenedReIDCoordinator"]
