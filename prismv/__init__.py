"""PRISM-V: Programmable Reasoning with Integrated Spatial Modules, with Retrieval-based
Verification.

A zero-shot spatial-reasoning framework in which an MLLM writes Python visual programs
against a spatial tool library, turning multi-view images into an explorable 3D scene, and
a lightweight verification layer (GeoVer) retrieves geometric evidence from that scene to
check the model's spatial hypotheses before the answer is committed.

Three stages, in order:

    perceive   reconstruct the scene and lift entity masks into 3D  (prismv.reconstruct_pipe,
               prismv.tools, prismv.geover.grounding)
    verify     retrieve evidence, check constraints, rerank         (prismv.geover)
    reason     execute the generated program, answer                (prismv.agent)

The package is importable without torch / VGGT / any API key: :mod:`prismv.geover` is pure
numpy, which is what makes the verification layer testable offline
(see ``tests/test_geover_synthetic.py``).
"""

from __future__ import annotations

__version__ = "0.1.0"

# GeoVer has no heavy dependencies, so it is always available.
from prismv.geover import (  # noqa: F401
    AnswerReranker,
    Assertion,
    GeoVer,
    GeoVerifier,
    GeometricEvidenceIndex,
    PREDICATES,
    RELATION_ALIASES,
    RerankResult,
    SceneGraph,
    VerificationReport,
    Verdict,
    build_scene_graph,
    relation_support_report,
)

# The perception/agent stack needs torch + model weights; expose it lazily so that
# `import prismv` and the offline tests keep working on a machine with neither.
_HEAVY = {
    "PRISM_V": "prismv.interface",
    "Agent": "prismv.interface",
    "Scene": "prismv.interface",
    "Reconstruction": "prismv.interface",
}


def __getattr__(name):
    if name in _HEAVY:
        import importlib

        mod = importlib.import_module(_HEAVY[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'prismv' has no attribute {name!r}")


def pipeline_summary() -> dict:
    """What the framework can check, and how much geometry each check consumes."""
    return {
        "stages": ["perceive", "verify", "reason"],
        "relations_verifiable": sorted(PREDICATES),
        "verification": "retrieval-based geometric evidence (GeoVer)",
        "requires_gpu_for": ["reconstruct", "novel_view_synthesis"],
        "pure_numpy": ["prismv.geover"],
    }


__all__ = [
    "GeoVer", "GeoVerifier", "GeometricEvidenceIndex", "SceneGraph", "VerificationReport",
    "RerankResult", "AnswerReranker", "Assertion", "Verdict", "PREDICATES",
    "RELATION_ALIASES", "build_scene_graph", "relation_support_report",
    "PRISM_V", "Agent", "Scene", "Reconstruction", "pipeline_summary", "__version__",
]
