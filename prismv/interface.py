"""PRISM-V spatial tool API.

``PRISM_V`` is the library the code agent programs against. It carries every perception
primitive from the inherited baseline pipeline (reconstruction, camera-motion description,
novel-view synthesis, pose algebra) and adds the second half of the pipeline: entity
grounding and retrieval-based geometric verification.

Two structural decisions:

**Heavy dependencies are imported lazily.** torch / open3d / VGGT weights are needed only
when a program actually calls ``reconstruct`` or ``synthesize_novel_view``. Keeping the
verification surface importable without them is what lets GeoVer be unit-tested on a plain
Python install, and it means a broken perception environment shows up as a failure of the
specific tool call rather than of the whole framework.

**Verification lives in the same API the agent writes programs against.** A verifier the
model can address can be asked for exactly the evidence a given question needs; a verifier
bolted on after the fact can only audit whatever the program happened to render. That is
the difference between ``PRISM_V.verify(scene, "right_of", "cup", "book", ref_view=1)``
returning the angular margin that decided it, and a rendered image returning nothing
checkable at all.
"""

from __future__ import annotations

import glob as _glob
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from prismv.geover import (
    GeoVer,
    GeoVerifier,
    GeometricEvidenceIndex,
    PREDICATES,
    SceneGraph,
    build_scene_graph,
    relation_support_report,
)
from prismv.geover.grounding import has_artefacts
from prismv.geover.hypothesis import Assertion

__all__ = ["Reconstruction", "Scene", "EntitySet", "PRISM_V", "PRISMV"]


# ------------------------------------------------------------------ data model
class Reconstruction:
    """The reconstructed scene: a point cloud plus the cameras that produced it.

    ``point_cloud`` is (M, 3) world-frame; ``extrinsics`` is (N, 3, 4) or (N, 4, 4) with
    ``extrinsics[i]`` mapping world->camera for view i (0-based, so image 1 is index 0).
    ``_processed_dir`` remembers where the on-disk artefacts live, which is what lets
    grounding find the per-view pointmap without the caller passing it every time.
    """

    def __init__(self, point_cloud, extrinsics, intrinsics, processed_dir: Optional[str] = None):
        self.point_cloud = point_cloud
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self._processed_dir = processed_dir
        self._raw_result: Optional[dict] = None


class Scene:
    """Input images, the question, and whatever perception has produced so far."""

    def __init__(self, path_to_images: Union[str, List[str]], question: str = "",
                 scene_id: Optional[str] = None):
        self.question = question
        self.scene_id = scene_id
        self.images = self._load_images(path_to_images)
        self.reconstruction: Optional[Reconstruction] = None
        self.code: Optional[str] = None
        self.visual_clue = None
        self.entities: Optional[SceneGraph] = None
        self.entity_provenance: Dict = {}

    def _load_images(self, path_to_images: Union[str, List[str]]) -> List[str]:
        if isinstance(path_to_images, str):
            if os.path.isdir(path_to_images):
                images: List[str] = []
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    images.extend(_glob.glob(os.path.join(path_to_images, ext)))
                return sorted(images)
            return [path_to_images]
        return list(path_to_images)


class EntitySet:
    """Grounded entities, plus the provenance of *how* they were grounded."""

    def __init__(self, graph: SceneGraph, provenance: Dict):
        self.graph = graph
        self.provenance = provenance

    @property
    def names(self) -> List[str]:
        return [e.name for e in self.graph.entities]

    def __len__(self):
        return len(self.graph.entities)

    def __iter__(self):
        return iter(self.graph.entities)

    def __bool__(self):
        return len(self.graph.entities) > 0

    def __repr__(self):
        return (f"EntitySet(n={len(self)}, grounding="
                f"{self.provenance.get('grounding_path')}, "
                f"extent={self.graph.extent():.3f})")


# ------------------------------------------------------------------ the API surface
class PRISM_V:
    """Spatial tools: perception primitives + retrieval-based verification (GeoVer)."""

    #: base directory where ``reconstruct_pipe`` wrote per-scene artefacts
    PROCESSED_BASE_DIR: Optional[str] = None

    # ---------------------------------------------------------------- perception
    @staticmethod
    def _resolve_processed_dir(scene: "Scene") -> Optional[str]:
        """Locate a scene's cached artefacts.

        Two layouts are supported because the batch pipeline writes ``BASE/<scene_id>/``
        while single-scene runs write straight into the output directory. Trying the
        per-scene subdirectory first and the base itself second keeps both working without
        a config flag.
        """
        base = PRISM_V.PROCESSED_BASE_DIR
        if not base:
            return None
        if scene.scene_id:
            cand = os.path.join(base, str(scene.scene_id))
            if has_artefacts(cand):
                return cand
        if has_artefacts(base):
            return base
        return None

    @staticmethod
    def reconstruct(scene: "Scene", processed_dir: Optional[str] = None) -> Reconstruction:
        """Reconstruct the scene from its images, reusing on-disk results when present.

        Lookup order: explicit ``processed_dir``, then ``PROCESSED_BASE_DIR/scene_id``, then
        live reconstruction. Reusing cached artefacts is what makes the reconstruction-quality
        sweep cheap: the same checkpoint is re-read under different verification settings
        instead of re-run, so the only thing varying between arms is the setting under test.
        """
        processed_dir = processed_dir or PRISM_V._resolve_processed_dir(scene)
        if processed_dir:
            loaded = _load_processed_scene(processed_dir)
            if loaded is not None:
                print(f"Loaded processed scene from: {processed_dir}")
                scene.reconstruction = loaded
                return loaded

        from prismv.tools.reconstruct import reconstruct_3d  # lazy: needs torch + weights

        result = reconstruct_3d(scene.images, scene_id=scene.scene_id)
        point_cloud = _as_numpy(result.get("points"))
        cameras = _as_numpy(result.get("cameras"))
        intrinsics = None
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            poses = metadata.get("camera_poses", {})
            if isinstance(poses, dict) and "intrinsic" in poses:
                intrinsics = np.array(poses["intrinsic"])
        recon = Reconstruction(point_cloud, cameras, intrinsics, processed_dir=processed_dir)
        recon._raw_result = result
        scene.reconstruction = recon
        return recon

    @staticmethod
    def describe_camera_motion(recon: Reconstruction):
        """Textual description of the camera trajectory across views."""
        from prismv.tools.camera_understanding import analyze_camera_trajectory

        return analyze_camera_trajectory(recon.extrinsics)

    @staticmethod
    def synthesize_novel_view(recon: Reconstruction, new_camera_pose, width=512, height=512,
                              out_path: Optional[str] = None):
        """Render the scene from a new viewpoint (3x4 or 4x4 world->camera matrix)."""
        from prismv.tools.novel_view_synthesis import novel_view_synthesis

        return novel_view_synthesis(recon, new_camera_pose, width, height, out_path)

    # pose algebra, unchanged signatures so previously generated programs keep running
    @staticmethod
    def _rotation_axis(recon: Optional[Reconstruction]):
        if recon is not None and getattr(recon, "extrinsics", None) is not None:
            from prismv.tools.novel_view_synthesis import average_look_at_directions

            ex = np.asarray(recon.extrinsics)
            if ex.ndim == 3:
                return average_look_at_directions(ex)
        return None

    @staticmethod
    def rotate_right(extrinsic, angle: Optional[float] = None,
                     recon: Optional[Reconstruction] = None):
        from prismv.tools.novel_view_synthesis import rotate_right as _fn

        axis = PRISM_V._rotation_axis(recon)
        return _fn(extrinsic, axis=axis) if angle is None else _fn(extrinsic, angle, axis=axis)

    @staticmethod
    def rotate_left(extrinsic, angle: Optional[float] = None,
                    recon: Optional[Reconstruction] = None):
        from prismv.tools.novel_view_synthesis import rotate_left as _fn

        axis = PRISM_V._rotation_axis(recon)
        return _fn(extrinsic, axis=axis) if angle is None else _fn(extrinsic, angle, axis=axis)

    @staticmethod
    def move_forward(extrinsic, distance: Optional[float] = None):
        from prismv.tools.novel_view_synthesis import move_forward as _fn

        return _fn(extrinsic) if distance is None else _fn(extrinsic, distance)

    @staticmethod
    def move_backward(extrinsic, distance: Optional[float] = None):
        from prismv.tools.novel_view_synthesis import move_backward as _fn

        return _fn(extrinsic) if distance is None else _fn(extrinsic, distance)

    @staticmethod
    def turn_around(extrinsic, recon: Optional[Reconstruction] = None):
        from prismv.tools.novel_view_synthesis import turn_around as _fn

        return _fn(extrinsic, axis=PRISM_V._rotation_axis(recon))

    # ---------------------------------------------------------------- grounding
    @staticmethod
    def ground_entities(scene: "Scene", entity_masks: Optional[Sequence[Dict]] = None,
                        processed_dir: Optional[str] = None, conf_thres: float = 0.0,
                        fallback_clusters: int = 0) -> EntitySet:
        """Lift 2D entity masks into 3D point sets; cluster on geometry if none exist.

        Always returns an EntitySet, including an empty one, so a generated program can
        branch on ``len(entities)`` instead of raising. When nothing could be grounded the
        verification calls downstream report abstention rather than inventing constraints --
        that distinction is the difference between a verifier and a second opinion.
        """
        processed_dir = processed_dir or getattr(scene.reconstruction, "_processed_dir", None) \
            or PRISM_V._resolve_processed_dir(scene)
        # Cached geometry first. If the artefacts already contain entity points or a
        # pointmap, reconstruction adds nothing that verification will use, and forcing it
        # would make the verification layer depend on a working VGGT install for no reason.
        perception_error = ""
        if scene.reconstruction is None and not has_artefacts(processed_dir):
            try:
                PRISM_V.reconstruct(scene)
            except ImportError as e:
                # The perception stack (torch / open3d / VGGT weights) is optional for
                # verification. Missing backend means *no new geometry*, which downstream
                # must show up as abstention -- not as a crash that takes the whole run with
                # it, and not as a silent zero-entity scene that looks like a model failure.
                perception_error = f"perception backend unavailable: {e}"
                print(f"[PRISM-V] {perception_error}; "
                      f"grounding with cached geometry only")
        graph, prov = build_scene_graph(
            processed_dir=processed_dir,
            reconstruction=scene.reconstruction,
            entity_masks=entity_masks,
            conf_thres=conf_thres,
            fallback_clusters=fallback_clusters,
            scene_id=scene.scene_id,
        )
        if perception_error:
            prov["perception_error"] = perception_error
        scene.entities = graph
        scene.entity_provenance = prov
        return EntitySet(graph, prov)

    @staticmethod
    def list_entities(scene: "Scene") -> List[str]:
        return [e.name for e in PRISM_V._graph(scene).entities]

    @staticmethod
    def entity_descriptions(scene: "Scene") -> List[Dict]:
        """Per-entity evidence summary: point counts, contributing views, density."""
        return [e.to_dict() for e in PRISM_V._graph(scene).entities]

    @staticmethod
    def scene_statistics(scene: "Scene") -> Dict:
        g = PRISM_V._graph(scene)
        return {
            "n_entities": len(g.entities),
            "n_views": None if g.extrinsics is None else int(len(g.extrinsics)),
            "extent": round(g.extent(), 5),
            "has_gravity": g.gravity is not None,
            "grounding": getattr(scene, "entity_provenance", {}),
            "entities": [e.to_dict() for e in g.entities],
        }

    # ---------------------------------------------------------------- verification
    @staticmethod
    def verify(scene: "Scene", relation: str, subject: str, obj: str,
               third: Optional[str] = None, ref_view: Optional[int] = None) -> Dict:
        """Check one explicit spatial claim against the reconstructed geometry.

        Returns the graded score, the signed margin, the deciding detail, and the entities
        actually used. ``status`` is one of supported / contradicted / undetermined --
        undetermined means the evidence does not license an answer, which is not the same
        thing as contradicting one.
        """
        idx = GeometricEvidenceIndex(PRISM_V._graph(scene))
        a = Assertion(relation=relation, subject=subject, object=obj, third=third,
                      ref_view=ref_view, source="program")
        check = GeoVerifier(idx).check(a)
        if check is None:
            return {"relation": relation, "status": "unsupported_relation",
                    "verifiable_relations": sorted(PREDICATES)}
        return check.to_dict()

    @staticmethod
    def verify_many(scene: "Scene", claims: Sequence[Dict]) -> List[Dict]:
        """Batch :meth:`verify`; each entry uses the same keyword names."""
        return [PRISM_V.verify(scene, c["relation"], c.get("subject", ""), c.get("object", ""),
                              c.get("third"), c.get("ref_view")) for c in claims]

    @staticmethod
    def check_options(scene: "Scene", question: str, options: Dict[str, str],
                      ref_view: Optional[int] = None, verification_weight: float = 1.0,
                      min_support: int = 1) -> Dict:
        """Verify a multiple-choice question end to end and rerank its options.

        ``ref_view=None`` infers the viewer from the question wording and echoes what it
        chose, so an auditor can see the layer did not silently pick a reference frame.
        """
        gv = GeoVer(graph=PRISM_V._graph(scene), verification_weight=verification_weight,
                    min_support=min_support)
        parsed = gv.parse(question, options or {})
        if ref_view is not None:
            for lst in (parsed.question_assertions, parsed.option_assertions):
                for a in lst:
                    a.ref_view = ref_view
            parsed.ref_view = ref_view
        letters = list((options or {}).keys())
        report = gv.verify(parsed, letters)
        rerank = gv.reranker.score(report, letters)
        return {
            "ref_view_inferred": parsed.ref_view,
            "n_assertions": len(parsed.option_assertions) + len(parsed.question_assertions),
            "counts": report.counts(),
            "option_coverage": report.option_coverage(),
            "ranked": [r.to_dict() for r in rerank.ranked],
            "answer": rerank.answer,
            "applied": rerank.applied,
            "note": rerank.note,
            "checks": [c.to_dict() for c in report.checks],
        }

    @staticmethod
    def rank_relations(scene: "Scene", anchor: str, candidates: Sequence[str],
                       relation: str, ref_view: Optional[int] = None) -> List[Dict]:
        """Order candidates by how strongly the geometry endorses ``relation`` to ``anchor``.

        ``anchor`` is the fixed referent, not the entity being located. For "which object is
        to the right of the book" the anchor is the book and every candidate must be the
        *subject* of the claim "candidate is right of book". Passing the anchor as the
        assertion subject instead evaluates the converse relation: on a single pair the two
        readings differ only in sign, so the mistake is invisible until candidates are
        actually ranked -- and then the ordering is systematically reversed.
        """
        g = PRISM_V._graph(scene)
        idx = GeometricEvidenceIndex(g)
        ver = GeoVerifier(idx)
        if relation not in PREDICATES:
            return []
        out = []
        for name in candidates:
            # candidate is the subject, anchor the referent -- see docstring
            a = Assertion(relation, subject=name, object=anchor, ref_view=ref_view,
                          source="program")
            check = ver.check(a)
            if check is None:
                out.append({"candidate": name, "score": 0.5,
                            "status": "unsupported_relation"})
                continue
            out.append({"candidate": name, "score": check.verdict.score,
                        "margin": check.verdict.margin, "status": check.status,
                        "detail": check.verdict.detail,
                        "entities_used": check.entities_used})
        return sorted(out, key=lambda d: -d["score"])

    @staticmethod
    def locate(scene: "Scene", description: str, top_k: int = 5) -> Dict:
        """Localise the entity that best matches a spatial description.

        Retrieval over the scene graph rather than over pixels: the answer is an entity with
        evidence attached, which is what makes localisation accuracy measurable at all.
        """
        g = PRISM_V._graph(scene)
        idx = GeometricEvidenceIndex(g)
        scored = []
        for e in g.entities:
            scored.append({"entity": e.name, "n_points": e.n_points,
                           "centroid": [round(v, 4) for v in e.centroid.tolist()],
                           "radius": round(e.radius(), 4)})
        view = detect_view(description)
        return {"description": description, "reference_view": view,
                "candidates": scored[:max(1, top_k)], "n_entities": len(g.entities)}

    @staticmethod
    def verifiable_relations() -> List[List[str]]:
        """Which relations this build can check, and what each check consumes. The analysis
        sweep reports this so relation coverage is stated rather than assumed."""
        return [[r, req] for r, req in relation_support_report()]

    # ---------------------------------------------------------------- internal
    @staticmethod
    def _graph(scene: "Scene") -> SceneGraph:
        g = getattr(scene, "entities", None)
        if g is not None:
            return g
        graph, prov = build_scene_graph(reconstruction=getattr(scene, "reconstruction", None),
                                        scene_id=getattr(scene, "scene_id", None))
        scene.entities = graph
        scene.entity_provenance = prov
        return graph


def detect_view(text: str) -> Optional[int]:
    """Which view a piece of text anchors its left/right language to (0-based)."""
    from prismv.geover.hypothesis import detect_reference_view

    return detect_reference_view(text)


def _as_numpy(x):
    if x is None:
        return None
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _load_processed_scene(processed_dir: str) -> Optional[Reconstruction]:
    """Read a scene previously written by ``reconstruct_pipe``.

    Supports both on-disk layouts: the pipeline's ``camera_matrices.npz`` + ``points.ply``,
    and the tool's ``cameras.npy`` + ``points3d.npy``. Returns None when the directory holds
    neither, which is the normal result for a scene that was never preprocessed.
    """
    if not os.path.isdir(processed_dir):
        return None
    point_cloud = extrinsics = intrinsics = None

    npz = os.path.join(processed_dir, "camera_matrices.npz")
    ply = os.path.join(processed_dir, "points.ply")
    meta = os.path.join(processed_dir, "processing_metadata.json")
    if os.path.exists(ply) and (os.path.exists(npz) or os.path.exists(meta)):
        try:
            import trimesh

            point_cloud = np.asarray(trimesh.load(ply).vertices)
        except Exception as e:
            print(f"[PRISM-V] could not read {ply}: {e}")
            return None
        if os.path.exists(npz):
            with np.load(npz) as d:
                extrinsics = d.get("extrinsic", None)
                intrinsics = d.get("intrinsic", None)
        elif os.path.exists(meta):
            import json

            with open(meta) as f:
                poses = json.load(f).get("camera_poses", {})
            if isinstance(poses, dict):
                extrinsics = np.array(poses["extrinsic"]) if "extrinsic" in poses else None
                intrinsics = np.array(poses["intrinsic"]) if "intrinsic" in poses else None
        return Reconstruction(point_cloud, extrinsics, intrinsics, processed_dir=processed_dir)

    cams = os.path.join(processed_dir, "cameras.npy")
    pts = os.path.join(processed_dir, "points3d.npy")
    if os.path.exists(pts):
        point_cloud = np.load(pts)
        extrinsics = np.load(cams) if os.path.exists(cams) else None
        return Reconstruction(point_cloud, extrinsics, intrinsics, processed_dir=processed_dir)
    return None


# Historical name is not re-exported: every program is generated against the
# documented PRISM_V surface, so no compatibility alias is maintained.
PRISMV = PRISM_V
