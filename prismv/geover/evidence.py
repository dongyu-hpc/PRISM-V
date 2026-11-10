"""GeoVer geometric evidence records and the retrieval index (PRISM-V).

The verification layer deliberately does NOT reason over the raw point cloud. Every check
operates on a small, addressable bundle of *retrieved* evidence: entity-level geometric
records plus the camera frames that define the relation being tested. Retrieval is what
keeps verification cheap and, more importantly, auditable -- each constraint can name the
exact evidence that decided it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _tokenize(text: str) -> List[str]:
    """Domain-agnostic tokenizer for lexical evidence retrieval (no external deps)."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"the", "a", "an", "is", "are", "to", "of", "in", "on", "at", "and", "or",
            "what", "which", "where", "with", "from", "there", "this", "that", "image",
            "images", "view", "views"}
    return [w for w in words if w not in stop and len(w) > 1]


@dataclass
class EntityEvidence:
    """One entity's geometry, addressable back to the pixels it came from.

    Attributes
    ----------
    name : canonical label used for retrieval.
    points : (M, 3) world-frame points. Kept (not just a centroid) so relations can be
        recomputed at any granularity -- centroid, PCA axes, angular footprint -- without
        re-lifting masks.
    views / pixel_counts : which views contributed and how many pixels each supplied.
        These make evidence *sufficiency* reportable, so verification can refuse rather
        than answer confidently off a 5-pixel reconstruction.
    """

    name: str
    points: np.ndarray
    views: List[int] = field(default_factory=list)
    pixel_counts: List[int] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 1.0

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float64).mean(axis=0)

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    def pca_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Principal axes (3x3, columns) and extents. Orientation evidence for asymmetric
        entities (a TV, a chair) that carry facing / alignment relations."""
        pts = np.asarray(self.points, dtype=np.float64)
        if len(pts) < 4:
            return np.eye(3), np.zeros(3)
        centered = pts - pts.mean(axis=0)
        cov = np.nan_to_num(np.cov(centered, rowvar=False))
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return np.eye(3), np.zeros(3)
        order = np.argsort(vals)[::-1]
        return vecs[:, order], np.sqrt(np.abs(vals[order]))

    def radius(self) -> float:
        """RMS spread around the centroid -- the local scale used to normalise margins."""
        pts = np.asarray(self.points, dtype=np.float64)
        if len(pts) == 0:
            return 0.0
        return float(np.sqrt(((pts - self.centroid) ** 2).sum(axis=1)).mean())

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_points": self.n_points,
            "centroid": [round(v, 5) for v in self.centroid.tolist()],
            "radius": round(self.radius(), 5),
            "views": self.views,
            "pixel_counts": self.pixel_counts,
            "confidence": round(float(self.confidence), 4),
            "description": self.description,
        }


@dataclass
class SceneGraph:
    """Entities + cameras for one scene: GeoVer's entire world model."""

    entities: List[EntityEvidence] = field(default_factory=list)
    extrinsics: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    gravity: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    scene_id: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    def by_name(self) -> Dict[str, EntityEvidence]:
        return {e.name: e for e in self.entities}

    def extent(self) -> float:
        """Median pairwise centroid distance among entities: the metric unit that turns
        raw metres into scale-free comparisons, so ``near`` means something in a 3 m room
        and a 30 m hall alike."""
        if len(self.entities) >= 2:
            cs = np.array([e.centroid for e in self.entities])
            d = np.linalg.norm(cs[:, None, :] - cs[None, :, :], axis=-1)
            iu = np.triu_indices(len(cs), 1)
            if len(iu[0]):
                med = float(np.median(d[iu]))
                if med > 1e-9:
                    return med
        if self.point_cloud is not None and len(self.point_cloud):
            pc = np.asarray(self.point_cloud, dtype=np.float64)
            pc = pc[np.isfinite(pc).all(axis=1)]
            if len(pc) > 1:
                return float(np.median(np.linalg.norm(pc - pc.mean(axis=0), axis=1)))
        return 1.0


@dataclass
class EvidenceBundle:
    """The minimal evidence actually retrieved for one assertion.

    Carrying the ``graph`` reference (not just the entities) is what lets a predicate
    resolve the reference camera frame and the scene scale, which are properties of the
    scene rather than of the two entities involved.
    """

    entities: Dict[str, Optional[EntityEvidence]]
    graph: SceneGraph
    ref_view: Optional[int] = None
    gravity: Optional[np.ndarray] = None
    reason: str = ""
    sufficient: bool = True

    def missing(self) -> List[str]:
        return [k for k, v in self.entities.items() if v is None and k]


class GeometricEvidenceIndex:
    """Retrieval over geometric evidence: lexical entity resolution, spatial
    neighbourhood queries, and per-assertion bundle assembly."""

    def __init__(self, graph: SceneGraph):
        self.graph = graph
        self._norm_names: Dict[str, EntityEvidence] = {}
        self._tokens: Dict[str, List[str]] = {}
        for e in graph.entities:
            for k in [e.name] + list(e.aliases):
                nk = self.normalize(k)
                if nk:
                    self._norm_names.setdefault(nk, e)
            self._tokens[self.normalize(e.name)] = _tokenize(e.name + " " + e.description)

    # ------------------------------------------------------------------ naming
    @staticmethod
    def normalize(name: str) -> str:
        if not name:
            return ""
        s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
        s = re.sub(r"\s+", " ", s).strip()
        # crude singularisation so "shoes" still matches "shoe"
        words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in s.split()]
        return " ".join(words)

    def find(self, name: str) -> Optional[EntityEvidence]:
        """Resolve a mentioned entity to a reconstructed entity.

        Exact normalised match first, then token overlap with a substring bonus. The 0.34
        floor is deliberately conservative: silently binding "lamp" to "table lamp frame"
        is fine, binding it to "sofa" is not.
        """
        nk = self.normalize(name)
        if not nk:
            return None
        if nk in self._norm_names:
            return self._norm_names[nk]

        qt = set(_tokenize(name))
        if not qt:
            # fall back to the raw normalised string as a single token
            qt = {nk}
        best, best_score = None, 0.0
        for cand, toks in self._tokens.items():
            ct = set(toks) if toks else {cand}
            inter = len(qt & ct)
            score = inter / max(len(qt), len(ct)) if inter else 0.0
            cjoined, qjoined = " ".join(sorted(ct)), " ".join(sorted(qt))
            if qjoined and qjoined in cjoined:
                score += 0.3
            elif cjoined and cjoined in qjoined:
                score += 0.3
            if score > best_score:
                best, best_score = cand, score
        return self._norm_names[best] if best_score >= 0.34 else None

    def nearest(self, entity: EntityEvidence, k: int = 4) -> List[Tuple[EntityEvidence, float]]:
        """Spatial neighbourhood retrieval -- the fallback when a question says "the
        object to the right" without naming which object."""
        if not self.graph.entities:
            return []
        cs = np.array([e.centroid for e in self.graph.entities])
        names = [e.name for e in self.graph.entities]
        d = np.linalg.norm(cs - entity.centroid, axis=1)
        out = []
        for i in np.argsort(d):
            if names[i] == entity.name:
                continue
            out.append((self.graph.entities[i], float(d[i])))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------- per-assertion fetch
    def bundle_for(self, assertion) -> EvidenceBundle:
        """Assemble exactly the evidence one assertion needs, and flag insufficiency."""
        ents: Dict[str, Optional[EntityEvidence]] = {}
        for slot in ("subject", "object", "third"):
            ref = getattr(assertion, slot, None)
            if not ref:
                continue
            ents[slot] = self.find(ref)
        missing = [s for s, v in ents.items() if v is None]
        if missing:
            return EvidenceBundle(ents, self.graph, getattr(assertion, "ref_view", None),
                                  self.graph.gravity,
                                  reason="unresolved: " + ",".join(missing),
                                  sufficient=False)
        return EvidenceBundle(ents, self.graph, getattr(assertion, "ref_view", None),
                              self.graph.gravity, reason="ok")

    def stats(self) -> Dict:
        return {
            "n_entities": len(self.graph.entities),
            "extent": round(self.graph.extent(), 5),
            "n_views": None if self.graph.extrinsics is None else int(len(self.graph.extrinsics)),
            "has_gravity": self.graph.gravity is not None,
            "min_points": min([e.n_points for e in self.graph.entities], default=0),
        }
