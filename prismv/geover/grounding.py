"""Entity grounding: lifting 2D entity masks into 3D point sets (PRISM-V).

This module supplies the geometric substrate GeoVer verifies against. The substrate was
missing upstream: the inherited interface referenced ``tool.segment`` but the module was
absent, and the reconstruction pipeline flattened the per-view pointmap into a single
``points.ply``, which destroys the (view, row, col) addressing a mask lift requires.

Two grounding paths, chosen by what is available:

``lift_masks``     exact, preferred. 2D masks + per-view pointmap -> 3D points.
``cluster_entities`` fallback. Unordered 3D clusters from geometry alone, labelled by
                   position only, so verification can still run (at reduced precision)
                   when no segmenter or no mask output is available.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prismv.geover.evidence import EntityEvidence, SceneGraph
from prismv.geover.predicates import estimate_gravity, infer_pose_convention


# --------------------------------------------------------------------- artefact loading
def load_pointmap(dirpath: str) -> Optional[Dict[str, np.ndarray]]:
    """Load the per-view pointmap artefact written by reconstruct_pipe (PRISM-V mode).

    Returns ``{pointmap, conf, rgb, extrinsic, intrinsic}`` or None. The absence of this
    artefact is the common case for checkpoints written before this stage existed, so
    callers must fall back rather than fail.
    """
    pm = os.path.join(dirpath, "pointmaps.npz")
    if not os.path.exists(pm):
        return None
    try:
        with np.load(pm, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"[grounding] failed to read {pm}: {e}")
        return None


def load_entities_file(dirpath: str) -> Optional[List[Dict]]:
    """Load a precomputed entity spec (2D masks or 3D points) if one was exported."""
    for cand in ("entities.json", "grounding.json"):
        p = os.path.join(dirpath, cand)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    obj = json.load(f)
                return obj.get("entities", obj if isinstance(obj, list) else None)
            except Exception as e:
                print(f"[grounding] failed to read {p}: {e}")
    return None


def load_entity_points(dirpath: str) -> List[EntityEvidence]:
    """Read entities that already carry 3D points, needing no pointmap or mesh reader.

    This is the injection path for externally-produced geometry: ground-truth simulators, a
    detector-plus-metric-depth pipeline, or a hand-authored synthetic scene. It exists
    because GeoVer's logic should be testable -- and usable -- independently of whether the
    VGGT stack happens to be installed.
    """
    specs = load_entities_file(dirpath)
    if not specs:
        return []
    out: List[EntityEvidence] = []
    for spec in specs:
        pts = spec.get("points")
        if pts is None and spec.get("points_file"):
            fp = spec["points_file"]
            if not os.path.isabs(fp):
                fp = os.path.join(dirpath, fp)
            if os.path.exists(fp):
                pts = np.load(fp) if fp.endswith(".npy") else np.loadtxt(fp, delimiter=",")
        if pts is None:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
            continue
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) == 0:
            continue
        out.append(EntityEvidence(
            name=str(spec.get("name", "")).strip(), points=arr,
            views=[int(v) for v in spec.get("views", [])],
            pixel_counts=[int(c) for c in spec.get("pixel_counts", [len(arr)])],
            aliases=[str(a) for a in spec.get("aliases", [])],
            description=str(spec.get("description", "")),
            confidence=float(spec.get("confidence", 1.0))))
    return [e for e in out if e.name]


def has_artefacts(dirpath: Optional[str]) -> bool:
    """True when a directory already holds something GeoVer can ground against.

    Checked before reconstruction so a scene with cached entity geometry never pays for a
    VGGT run it does not need.
    """
    if not dirpath or not os.path.isdir(dirpath):
        return False
    for name in ("entities.json", "grounding.json", "pointmaps.npz", "camera_matrices.npz",
                 "points.ply", "points3d.npy"):
        if os.path.exists(os.path.join(dirpath, name)):
            return True
    return False


# --------------------------------------------------------------------- mask -> 3D
def lift_masks(entity_masks: Sequence[Dict], pointmap: np.ndarray,
               conf: Optional[np.ndarray] = None,
               conf_thres: float = 0.0,
               max_pixels_per_entity: int = 60000) -> List[EntityEvidence]:
    """Lift per-view 2D masks into world-frame 3D point sets.

    ``entity_masks`` entries: ``{name, aliases?, description?, masks: [{view, mask}]}``
    where ``mask`` is a boolean array at pointmap resolution, or ``{view, boxes}`` for
    axis-aligned boxes (a cheaper contract that still yields correct centroids).

    Points below ``conf_thres`` are dropped, which is the same quality knob the analysis
    sweep uses, so grounding and reconstruction quality stay comparable across ablations.
    """
    n_views, H, W, _ = pointmap.shape
    out: List[EntityEvidence] = []
    for spec in entity_masks or []:
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        chunks: List[np.ndarray] = []
        views_used: List[int] = []
        counts: List[int] = []
        for m in spec.get("masks", []):
            v = int(m.get("view", m.get("image_index", 0)))
            if not (0 <= v < n_views):
                continue
            pm_view = pointmap[v]
            if "mask" in m:
                mask = np.asarray(m["mask"])
                if mask.dtype != bool:
                    mask = mask.astype(bool)
                if mask.shape != (H, W):
                    # tolerate full-resolution masks by nearest-neighbour resampling
                    from numpy import indices
                    rr = (np.linspace(0, mask.shape[0] - 1, H)).astype(int)
                    cc = (np.linspace(0, mask.shape[1] - 1, W)).astype(int)
                    mask = mask[rr][:, cc].astype(bool)
            elif "boxes" in m:
                mask = np.zeros((H, W), dtype=bool)
                for box in m["boxes"]:
                    x0, y0, x1, y1 = [int(round(float(b))) for b in box[:4]]
                    x0, x1 = max(0, min(x0, W - 1)), max(0, min(x1, W - 1))
                    y0, y1 = max(0, min(y0, H - 1)), max(0, min(y1, H - 1))
                    if x1 > x0 and y1 > y0:
                        mask[y0:y1 + 1, x0:x1 + 1] = True
            else:
                continue
            sel = pm_view[mask]
            sel = sel[np.isfinite(sel).all(axis=1)]
            if conf is not None and conf_thres > 0:
                cm = conf[v][mask]
                keep = np.asarray(cm) >= conf_thres
                sel = sel[keep[: len(sel)]] if len(keep) == len(sel) else sel
            if len(sel) == 0:
                continue
            if len(sel) > max_pixels_per_entity:
                idx = np.random.RandomState(0).choice(len(sel), max_pixels_per_entity, replace=False)
                sel = sel[np.sort(idx)]
            chunks.append(sel)
            views_used.append(v)
            counts.append(int(len(sel)))
        if not chunks:
            continue
        pts = np.concatenate(chunks, axis=0)
        mean_conf = float(np.mean(counts)) / max(1, H * W)
        out.append(EntityEvidence(
            name=name,
            points=pts,
            views=views_used,
            pixel_counts=counts,
            aliases=[str(a) for a in spec.get("aliases", [])],
            description=str(spec.get("description", "")),
            # evidence density: an entity covered by 12 pixels in one view should not
            # carry the same authority as one seen from four views.
            confidence=float(min(1.0, 0.35 + 0.25 * len(views_used) + 4.0 * mean_conf)),
        ))
    return out


# --------------------------------------------------------------------- geometry-only fallback
def cluster_entities(point_cloud: np.ndarray, extrinsics: Optional[np.ndarray] = None,
                    k: int = 6, seed: int = 0) -> List[EntityEvidence]:
    """Position-labelled 3D clusters when no 2D masks exist.

    These are *not* real entities -- they are place holders GeoVer can still reason about
    for relations like "which cluster is nearest the viewer". The names are geometric
    ("cluster_0"), so the hypothesis parser will normally refuse to bind them, which is the
    correct outcome: unverifiable should not become verified.
    """
    pts = np.asarray(point_cloud, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < max(8, k * 4):
        return []
    k = int(min(k, max(2, len(pts) // 8)))
    rng = np.random.RandomState(seed)
    centers = pts[rng.choice(len(pts), k, replace=False)].copy()
    for _ in range(24):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        new = centers.copy()
        for j in range(k):
            m = assign == j
            if m.any():
                new[j] = pts[m].mean(axis=0)
        if np.allclose(new, centers, atol=1e-6):
            centers = new
            break
        centers = new
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(d, axis=1)
    out: List[EntityEvidence] = []
    for j in range(k):
        sel = pts[assign == j]
        if len(sel) < 8:
            continue
        out.append(EntityEvidence(name=f"cluster_{j}", points=sel, views=[],
                                  pixel_counts=[int(len(sel))], confidence=0.25,
                                  description="geometry-only cluster, unlabelled"))
    return out


# --------------------------------------------------------------------- scene assembly
def load_camera_spec(dirpath: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read camera matrices without needing a mesh reader.

    ``camera_matrices.npz`` is plain numpy, so extrinsics are available even on installs with
    no trimesh -- which matters because gravity estimation and every view-relative predicate
    depend on the cameras, not on the point cloud.
    """
    if not dirpath or not os.path.isdir(dirpath):
        return None, None
    npz = os.path.join(dirpath, "camera_matrices.npz")
    if os.path.exists(npz):
        try:
            with np.load(npz, allow_pickle=False) as d:
                return (d["extrinsic"] if "extrinsic" in d.files else None,
                        d["intrinsic"] if "intrinsic" in d.files else None)
        except Exception as e:
            print(f"[grounding] could not read {npz}: {e}")
    for cand in ("entities.json", "grounding.json"):
        p = os.path.join(dirpath, cand)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    ex = obj.get("extrinsics", obj.get("extrinsic"))
                    it = obj.get("intrinsics", obj.get("intrinsic"))
                    if ex is not None:
                        return np.asarray(ex, dtype=np.float64), (
                            np.asarray(it, dtype=np.float64) if it is not None else None)
            except Exception:
                pass
    return None, None


def build_scene_graph(processed_dir: Optional[str] = None,
                      reconstruction=None,
                      entity_masks: Optional[Sequence[Dict]] = None,
                      extrinsics: Optional[np.ndarray] = None,
                      intrinsics: Optional[np.ndarray] = None,
                      point_cloud: Optional[np.ndarray] = None,
                      conf_thres: float = 0.0,
                      scene_id: Optional[str] = None,
                      fallback_clusters: int = 0) -> Tuple[SceneGraph, Dict]:
    """Assemble a SceneGraph from whatever artefacts exist, and report how it was built.

    The returned provenance dict matters for the analysis sweep: verification quality
    depends on which grounding path fired, and a paper that hides that distinction cannot
    explain why GeoVer helps on some scenes and not others.
    """
    diag: Dict[str, object] = {"processed_dir": processed_dir, "grounding_path": "none"}
    pm = None
    if processed_dir:
        pm = load_pointmap(processed_dir)
    if extrinsics is None and pm is not None and "extrinsic" in pm:
        extrinsics = pm["extrinsic"]
    if intrinsics is None and pm is not None and "intrinsic" in pm:
        intrinsics = pm["intrinsic"]
    # view-relative predicates and gravity estimation depend on the cameras, not the cloud,
    # and camera_matrices.npz is plain numpy -- so cameras must resolve even when the mesh
    # reader is absent and no reconstruction object was passed in.
    if extrinsics is None and processed_dir:
        extrinsics, intrinsics = load_camera_spec(processed_dir)
    if point_cloud is None and reconstruction is not None:
        point_cloud = getattr(reconstruction, "point_cloud", None)
        extrinsics = getattr(reconstruction, "extrinsics", None) if extrinsics is None else extrinsics
        intrinsics = getattr(reconstruction, "intrinsics", None) if intrinsics is None else intrinsics
        if pm is None:
            pd = getattr(reconstruction, "_processed_dir", None)
            if pd:
                pm = load_pointmap(pd)
    if point_cloud is None and pm is not None:
        point_cloud = pm["pointmap"].reshape(-1, 3)

    entities: List[EntityEvidence] = []
    if entity_masks is None and processed_dir:
        entity_masks = load_entities_file(processed_dir)

    # Order matters. Already-3D entity points are the strongest evidence available (exact,
    # no re-projection loss), so they win over lifting masks through a pointmap, which in
    # turn wins over unsupervised clustering. Recording which path fired is what lets the
    # analysis attribute a verification gain or loss to the grounding stage rather than to
    # the checker.
    if processed_dir:
        entities = load_entity_points(processed_dir)
        if entities:
            diag["grounding_path"] = "entity_points"
    if not entities and pm is not None and entity_masks:
        entities = lift_masks(entity_masks, pm["pointmap"], pm.get("conf"),
                              conf_thres=conf_thres)
        if entities:
            diag["grounding_path"] = "mask_lift"
    if not entities and entity_masks and pm is None:
        # masks without a pointmap cannot be lifted; say so instead of silently clustering
        diag["grounding_path"] = "masks_without_pointmap"
    if not entities and fallback_clusters and point_cloud is not None:
        entities = cluster_entities(point_cloud, extrinsics, k=fallback_clusters)
        if entities:
            diag["grounding_path"] = "cluster_fallback"
    if not entities and "grounding_path" not in diag:
        diag["grounding_path"] = "none"

    if extrinsics is not None and point_cloud is not None:
        diag["pose_convention_world_to_cam"] = bool(infer_pose_convention(extrinsics, point_cloud))
    gravity = estimate_gravity(extrinsics) if extrinsics is not None else None
    diag["gravity_estimated"] = gravity is not None
    diag["n_entities_grounded"] = len(entities)

    graph = SceneGraph(entities=entities, extrinsics=extrinsics, intrinsics=intrinsics,
                       gravity=gravity, point_cloud=point_cloud, scene_id=scene_id,
                       meta=diag)
    return graph, diag
