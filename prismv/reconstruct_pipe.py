"""Thin re-export of the repository-root ``reconstruct_pipe`` module.

The perception pipeline lives at the repo root because VGGT's weight loading resolves
paths relative to that file; duplicating it inside the package would fork those assumptions
and silently break weight discovery. This shim keeps a single authoritative copy while
still giving ``prismv`` a clean import surface.
"""

from reconstruct_pipe import *  # noqa: F401,F403
from reconstruct_pipe import (  # noqa: F401
    ReconstructionPipeline,
    process_scene,
)

try:  # module-level entry point, kept importable for scripts
    from reconstruct_pipe import main  # noqa: F401
except Exception:  # pragma: no cover - depends on the root module's CLI surface
    pass
