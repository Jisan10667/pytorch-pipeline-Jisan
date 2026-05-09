"""Backward-compatible utilities module.

The original project tab used ``utills.py``. New code can import from
``going_moduler.utils``; this file re-exports the same helpers.
"""

from going_moduler.utils import *  # noqa: F401,F403
