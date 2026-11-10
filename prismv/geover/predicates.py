"""Geometric predicate library for GeoVer (PRISM-V).

Three design decisions carry most of the value here:

1. **Reference frames are explicit.** "left of" is meaningless without a viewer, and
   spatial hallucination in MLLMs concentrates exactly there. Every directional predicate
   takes a reference camera and refuses to answer when none is available.
2. **Verticality is estimated, not assumed.** VGGT output lives in an arbitrary world
   frame with no guarantee that any axis is up, so ``above``/``below`` use a gravity axis
   inferred from the camera poses instead of a hard-coded coordinate.
3. **Margins, not booleans.** Each predicate returns a graded score from a
   scale-normalised geometric margin, which is what lets the reranker prefer
   *better-supported* options rather than merely filtering contradicted ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# An entity reconstructed from fewer than this many lifted points is not geometrically
# trustworthy; predicates return "undetermined" rather than a guess.
MIN_POINTS = 24


# --------------------------------------------------------------------- data model
@dataclass
class Verdict:
    """Outcome of one predicate evaluation."""

    holds: bool
    score: float          # [0, 1]: degree to which the relation holds
    margin: float         # signed, scale-normalised geometric evidence
    undetermined: bool = False
    detail: str = ""

    def to_dict(self) -> Dict:
        return {"holds": self.holds, "score": round(self.score, 4),
                "margin": round(self.margin, 4), "undetermined": self.undetermined,
                "detail": self.detail}


UNKNOWN = Verdict(holds=False, score=0.5, margin=0.0, undetermined=True,
                  detail="insufficient evidence")


def _sigmoid(x: float, scale: float = 1.0) -> float:
    z = float(np.clip(x / max(scale, 1e-9), -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-z)))


# ------------------------------------------------------------------ pose utilities
def as_world_to_cam(extrinsic) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise a (3, 4) or (4, 4) extrinsic into (R, t) with R 3x3."""
    e = np.asarray(extrinsic, dtype=np.float64)
    if e.shape == (4, 4):
        return e[:3, :3], e[:3, 3]
    if e.shape == (3, 4):
        return e[:, :3], e[:, 3]
    raise ValueError(f"unsupported extrinsic shape {e.shape}")


def camera_frame(extrinsic) -> Dict[str, np.ndarray]:
    """The camera axes expressed in world coordinates (OpenCV convention).

    ``right`` / ``down`` / ``forward`` are R^T e_i, because the stored matrix maps
    world->camera; the world-space camera centre is -R^T t.
    """
    R, t = as_world_to_cam(extrinsic)
    origin = -R.T @ t
    return {
        "origin": origin,
        "right": R.T @ np.array([1.0, 0.0, 0.0]),
        "down": R.T @ np.array([0.0, 1.0, 0.0]),
        "forward": R.T @ np.array([0.0, 0.0, 1.0]),
    }


def estimate_gravity(extrinsics) -> Optional[np.ndarray]:
    """Infer the world-space down direction from camera poses.

    Benchmark captures are handheld upright shots, so the mean of the per-camera ``down``
    axes is a robust gravity estimate. Deriving it from the cameras rather than trusting a
    fixed world axis is what makes above/below meaningful on unnormalised reconstructions.
    """
    if extrinsics is None or len(extrinsics) == 0:
        return None
    acc = np.zeros(3)
    n = 0
    for e in extrinsics:
        try:
            d = camera_frame(e)["down"]
        except ValueError:
            continue
        if np.isfinite(d).all() and np.linalg.norm(d) > 1e-8:
            acc += d / np.linalg.norm(d)
            n += 1
    if n == 0:
        return None
    g = acc / n
    ng = np.linalg.norm(g)
    return g / ng if ng > 1e-8 else None


def infer_pose_convention(extrinsics, points) -> bool:
    """Decide world->camera vs camera->world by testing which reading puts the
    reconstructed points in front of the cameras (positive optical depth).

    GeoVer's directional predicates are meaningless under the wrong convention, and this
    cannot be read off the file format alone, so it is inferred and then asserted.
    """
    if extrinsics is None or points is None or len(points) == 0:
        return True
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return True
    sample = pts[:: max(1, len(pts) // 500)]

    def positive_rate(world_to_cam: bool) -> float:
        good = total = 0
        for e in extrinsics:
            ee = np.asarray(e, dtype=np.float64)
            if ee.shape == (4, 4):
                ee = ee[:3, :4]
            if ee.shape != (3, 4):
                continue
            R, t = ee[:, :3], ee[:, 3]
            if not world_to_cam:
                R, t = R.T, -R.T @ t
            z = (R @ sample.T).T[:, 2] + t[2]
            total += len(z)
            good += int((z > 0).sum())
        return good / total if total else 0.0

    return positive_rate(True) >= positive_rate(False)


def _entity_ok(e) -> bool:
    return e is not None and getattr(e, "n_points", 0) >= MIN_POINTS


def _ref_camera(bundle):
    """Resolve the reference camera frame, or None if the assertion lacks a viewer."""
    frames = bundle.graph.extrinsics
    if frames is None or bundle.ref_view is None:
        return None
    if bundle.ref_view >= len(frames):
        return None
    try:
        return camera_frame(frames[bundle.ref_view])
    except ValueError:
        return None


# --------------------------------------------------------------------- predicates
def left_of(a, b, bundle) -> Verdict:
    return _lateral(a, b, bundle, sign=-1.0, name="left_of")


def right_of(a, b, bundle) -> Verdict:
    return _lateral(a, b, bundle, sign=+1.0, name="right_of")


def _lateral(a, b, bundle, sign: float, name: str) -> Verdict:
    """Lateral relation judged as an *angular* offset in the reference camera's image plane.

    Angular rather than metric: two entities at very different depths can share a lateral
    position and still look nothing alike in the image. Depth-independence is the whole
    point of phrasing it in radians.
    """
    if not (_entity_ok(a) and _entity_ok(b)):
        return UNKNOWN
    if bundle.ref_view is None:
        return Verdict(False, 0.5, 0.0, True,
                       "lateral relation needs an explicit reference view")
    cam = _ref_camera(bundle)
    if cam is None:
        return UNKNOWN
    origin, axis = cam["origin"], cam["right"]
    ca, cb = a.centroid, b.centroid
    na = np.linalg.norm(ca - origin)
    nb = np.linalg.norm(cb - origin)
    if min(na, nb) < 1e-9:
        return UNKNOWN
    ang_a = float((axis @ (ca - origin)) / na)
    ang_b = float((axis @ (cb - origin)) / nb)
    margin = sign * (ang_a - ang_b)
    score = _sigmoid(margin, 0.05)
    return Verdict(score > 0.5, score, margin, False,
                   f"{name} @view{bundle.ref_view}: angular offset {margin:+.4f} rad")


def closer_than(a, b, bundle) -> Verdict:
    return _metric_pair(a, b, bundle, sign=-1.0, name="closer_than")


def farther_than(a, b, bundle) -> Verdict:
    return _metric_pair(a, b, bundle, sign=+1.0, name="farther_than")


def _metric_pair(a, b, bundle, sign: float, name: str) -> Verdict:
    """Distance comparison from the viewer when a reference view exists, else from the
    reconstruction's own centre (so the predicate still works on text-only input)."""
    if not (_entity_ok(a) and _entity_ok(b)):
        return UNKNOWN
    origin = None
    cam = _ref_camera(bundle)
    if cam is not None:
        origin = cam["origin"]
    elif bundle.graph.point_cloud is not None and len(bundle.graph.point_cloud):
        pc = np.asarray(bundle.graph.point_cloud, dtype=np.float64)
        pc = pc[np.isfinite(pc).all(axis=1)]
        if len(pc):
            origin = pc.mean(axis=0)
    if origin is None:
        return UNKNOWN
    da = float(np.linalg.norm(a.centroid - origin))
    db = float(np.linalg.norm(b.centroid - origin))
    ext = max(bundle.graph.extent(), 1e-6)
    margin = sign * (da - db) / ext
    score = _sigmoid(margin, 0.05)
    return Verdict(score > 0.5, score, margin, False,
                   f"{name}: dist {da:.3f} vs {db:.3f} (extent {ext:.3f})")


def above(a, b, bundle) -> Verdict:
    return _vertical(a, b, bundle, sign=-1.0, name="above")


def below(a, b, bundle) -> Verdict:
    return _vertical(a, b, bundle, sign=+1.0, name="below")


def _vertical(a, b, bundle, sign: float, name: str) -> Verdict:
    """Vertical relation along the *estimated* gravity axis, with a body-scale dead zone.

    The dead zone matters: centroids 3 cm apart vertically are not meaningfully
    above/below each other, and emitting a confident verdict there is exactly the kind of
    spurious constraint that corrupts reranking.
    """
    if not (_entity_ok(a) and _entity_ok(b)):
        return UNKNOWN
    g = bundle.gravity
    if g is None:
        return Verdict(False, 0.5, 0.0, True, "no gravity axis estimated from camera poses")
    delta = float((a.centroid - b.centroid) @ np.asarray(g, dtype=np.float64))
    scale = max(a.radius() + b.radius(), 1e-3)
    signed = sign * delta / scale
    score = _sigmoid(signed, 0.5)
    if abs(delta) < 0.15 * scale:
        return Verdict(score > 0.5, score, signed, True,
                       f"{name}: vertical gap {delta:.3f} inside entity scale -> undetermined")
    return Verdict(score > 0.5, score, signed, False,
                   f"{name}: vertical gap {delta:.3f} (scale {scale:.3f})")


def near(a, b, bundle, threshold: float = 0.6) -> Verdict:
    if not (_entity_ok(a) and _entity_ok(b)):
        return UNKNOWN
    ext = max(bundle.graph.extent(), 1e-6)
    d = float(np.linalg.norm(a.centroid - b.centroid)) / ext
    margin = threshold - d
    score = _sigmoid(margin, 0.15)
    return Verdict(score > 0.5, score, margin, False,
                   f"near: normalised gap {d:.3f} vs threshold {threshold}")


def far_from(a, b, bundle, threshold: float = 1.0) -> Verdict:
    v = near(a, b, bundle, threshold)
    return Verdict(not v.holds, 1.0 - v.score, -v.margin, v.undetermined,
                   "far_from: " + v.detail)


def between(a, b, bundle) -> Verdict:
    """``subject`` lies between ``object`` and the assertion's third entity."""
    c = bundle.entities.get("third")
    if not (_entity_ok(a) and _entity_ok(b) and _entity_ok(c)):
        return UNKNOWN
    cam = _ref_camera(bundle)
    if cam is None:
        return UNKNOWN
    origin, axis = cam["origin"], cam["right"]

    def lateral_pos(e):
        v = e.centroid - origin
        return float((axis @ v) / max(np.linalg.norm(v), 1e-9))

    pa, pb, pc = lateral_pos(a), lateral_pos(b), lateral_pos(c)
    lo, hi = min(pb, pc), max(pb, pc)
    width = max(hi - lo, 1e-9)
    inside = (pa - lo) / width              # in [0, 1] when laterally between
    margin = 0.5 - abs(inside - 0.5)        # maximal when exactly mid-way
    score = _sigmoid(margin, 0.12)
    in_range = -0.15 <= inside <= 1.15
    return Verdict(in_range and score > 0.5, score, margin, False,
                   f"between: interpolation {inside:.3f} across [{lo:+.3f},{hi:+.3f}]")


def occludes(a, b, bundle) -> Verdict:
    """Visibility test. Occlusion is a *conjunction*: the occluder must be nearer AND
    angularly cover the target.

    Both terms are normalised to the same "positive means the condition holds" convention
    and aggregated with ``min``, so failing either one makes the whole claim fail. A signed
    margin with a depth-based sign flip looks equivalent but is not: an entity that is both
    behind the target and too small to cover it ends up with two negatives multiplying into
    a positive, and the layer would then certify a geometrically impossible occlusion.
    """
    if not (_entity_ok(a) and _entity_ok(b)):
        return UNKNOWN
    cam = _ref_camera(bundle)
    if cam is None:
        return UNKNOWN
    origin = cam["origin"]

    def depth_and_dir(e):
        v = e.centroid - origin
        n = max(float(np.linalg.norm(v)), 1e-9)
        return n, v / n

    da, ua = depth_and_dir(a)
    db, ub = depth_and_dir(b)
    if da <= 0 or db <= 0:
        return UNKNOWN
    ang = float(np.arccos(np.clip(ua @ ub, -1.0, 1.0)))
    # RMS extent as the angular-radius proxy: exact enough for "does this block that",
    # and it degrades conservatively on elongated entities rather than over-claiming.
    angular_radius = float(np.arctan2(a.radius(), da))
    depth_margin = (db - da) / max(db, 1e-6)          # >0 iff a is nearer than b
    overlap_margin = (angular_radius - ang) / max(angular_radius, 1e-6)  # >0 iff covered
    margin = min(depth_margin, overlap_margin)
    score = _sigmoid(margin, 0.3)
    return Verdict(score > 0.5, score, margin, False,
                   f"occludes: depth margin {depth_margin:+.3f}, angular overlap "
                   f"{overlap_margin:+.3f} (sep {ang:.3f} rad vs radius "
                   f"{angular_radius:.3f} rad) -> {margin:+.3f}")


def faces_toward(a, b, bundle) -> Verdict:
    """Principal axis as a facing proxy (entity-level orientation relations)."""
    if not (_entity_ok(a) and _entity_ok(b)):
        return UNKNOWN
    if a.n_points < max(60, MIN_POINTS * 2):
        return UNKNOWN
    axes, extents = a.pca_axes()
    principal = axes[:, 0]
    target = b.centroid - a.centroid
    tn = float(np.linalg.norm(target))
    if tn < 1e-9 or extents[0] < 1e-6:
        return UNKNOWN
    cos = float(np.clip(principal @ (target / tn), -1.0, 1.0))
    # A flat entity's principal axis runs *along* its face, so "facing the target" shows up
    # as the axis being collinear with the bearing, not aligned with it in sign.
    margin = abs(cos) - 0.5
    score = _sigmoid(margin, 0.2)
    return Verdict(score > 0.5, score, margin, False,
                   f"faces_toward: |cos| between PCA axis and bearing = {abs(cos):.3f}")


def same_side(a, b, bundle) -> Verdict:
    """Do two entities fall on the same lateral side of the reference view's centre?"""
    if not (_entity_ok(a) and _entity_ok(b)):
        return UNKNOWN
    cam = _ref_camera(bundle)
    if cam is None:
        return UNKNOWN
    origin, axis = cam["origin"], cam["right"]

    def side(e):
        return float(np.sign(axis @ (e.centroid - origin)))

    sa, sb = side(a), side(b)
    same = (sa == sb) and sa != 0.0
    return Verdict(same, 1.0 if same else 0.0, 1.0 if same else -1.0, False,
                   "same_side: " + ("agree" if same else "disagree"))


# ------------------------------------------------------------ vocabulary / registry
# Surface phrasings in questions and options, mapped onto predicates. Kept as data so the
# analysis module can report coverage -- which spatial relations GeoVer can actually check.
RELATION_ALIASES: Dict[str, List[str]] = {
    "right_of": ["to the right of", "on the right of", "right of", "right side of",
                 "toward the right from", "is right of"],
    "left_of": ["to the left of", "on the left of", "left of", "left side of",
                "toward the left from", "is left of"],
    "closer_than": ["closer to", "nearer to", "nearest to", "closest to", "closer than",
                    "in front of", "nearest"],
    "farther_than": ["farther from", "further from", "farthest from", "behind",
                     "far from", "farthest", "more distant"],
    "above": ["above", "over", "on top of", "higher than", "upper"],
    "below": ["below", "under", "beneath", "lower than", "bottom"],
    "near": ["next to", "beside", "adjacent to", "near the", "close to the"],
    "between": ["between", "amid", "in the middle of"],
    "occludes": ["blocks", "occludes", "hides", "obscures"],
    "faces_toward": ["facing", "pointing at", "aims at", "looks toward"],
    "same_side": ["same side", "both to the"],
}

PREDICATES = {
    "left_of": left_of,
    "right_of": right_of,
    "closer_than": closer_than,
    "farther_than": farther_than,
    "above": above,
    "below": below,
    "near": near,
    "far_from": far_from,
    "between": between,
    "occludes": occludes,
    "faces_toward": faces_toward,
    "same_side": same_side,
}

# Relations that cannot be evaluated without an explicit viewer. Surfacing this set lets
# the pipeline log "needs reference view" instead of silently falling back to the world
# frame, which is where most of this benchmark family's geometry errors originate.
VIEW_RELATIVE = {"left_of", "right_of", "closer_than", "farther_than", "between",
                 "occludes", "same_side"}
GRAVITY_RELATIVE = {"above", "below"}


def relation_support_report() -> List[Tuple[str, str]]:
    """(relation, requirement) pairs -- input to the tool-coverage analysis."""
    rows = []
    for r in PREDICATES:
        if r in VIEW_RELATIVE:
            rows.append((r, "reference camera"))
        elif r in GRAVITY_RELATIVE:
            rows.append((r, "estimated gravity"))
        else:
            rows.append((r, "entity centroids only"))
    return rows
