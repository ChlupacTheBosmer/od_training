"""Legacy compatibility wrapper for runtime config utilities.

New code should import from ``od_training.utility.runtime_config``.
This module is kept so older imports continue to work during migration.
"""

from .od_training.utility.runtime_config import *  # noqa: F401,F403
