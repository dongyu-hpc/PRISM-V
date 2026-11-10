"""Evaluation runner: the task x condition x verification ablation matrix.

The matrix is the point of this module. A single accuracy number on one benchmark cannot
support PRISM-V's claims -- that a tool-built 3D scene beats 2D images, that
verification adds to that rather than replacing it -- and each of those needs a different
cell filled by the *same* scenes. So the runner fixes the question set and varies only the
declared factors, and it refuses to compare cells computed over different subsets.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from prismv.eval import metrics
from prismv.eval.datasets import task_coverage
from prismv.interface import PRISM_V
from prismv.pipeline import Condition, PRISMVPipeline


@dataclass
class Cell:
    """One (task, condition, verification) setting over one question set."""

    task: str
    condition: str
    used_geover: bool
    results: List[Dict] = field(default_factory=list)
    wallclock_s: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.task}/{self.condition}/geover={'on' if self.used_geover else 'off'}"

    def summary(self) -> Dict:
        return {
            "task": self.task, "condition": self.condition, "used_geover": self.used_geover,
            "n": len(self.results), "wallclock_s": round(self.wallclock_s, 2),
            "answer": metrics.answer_accuracy(self.results),
            "health": metrics.pipeline_health(self.results),
            "effect": metrics.verification_effect(self.results),
            "grounding": metrics.grounding_quality(self.results),
            "relations": metrics.relation_level_accuracy(self.results),
        }


class EvalRunner:
    """Runs PRISM-V over a record set for every declared cell of the ablation matrix."""

    def __init__(self, pipeline: PRISMVPipeline, records: Optional[Sequence[Dict]] = None,
                 conditions: Sequence[str] = Condition.ALL,
                 verification_states: Sequence[bool] = (False, True),
                 tasks: Sequence[str] = ("qa",),
                 progress_every: int = 20, limit: Optional[int] = None):
        self.pipeline = pipeline
        self.records = list(records or [])
        if limit:
            self.records = self.records[:limit]
        self.conditions = list(conditions)
        self.verification_states = list(verification_states)
        self.tasks = list(tasks)
        self.progress_every = progress_every
        self.cells: List[Cell] = []

    # ------------------------------------------------------------------- per-task driver
    def _run_record(self, rec: Dict, task: str, condition: str,
                    used_geover: bool) -> Dict:
        if task == "qa":
            # must be serialised here: the caller only appends dict results, so returning the
            # PipelineResult object itself would drop every QA item without a word of warning
            return self.pipeline.run(
                question=rec["question"], options=rec.get("options") or {},
                images=rec.get("images"), condition=condition, used_geover=used_geover,
                scene_id=rec.get("scene_id"), expected_answer=rec.get("gt_answer"),
                entity_masks=rec.get("entity_masks")).to_dict()
        if task == "localize":
            return self._run_localization(rec, condition, used_geover)
        if task == "describe":
            return self._run_description(rec, condition, used_geover)
        raise ValueError(f"unknown task {task!r}")

    def _run_localization(self, rec: Dict, condition: str, used_geover: bool) -> Dict:
        """Localisation: rank candidate entities by geometric support for the described
        relation, then compare the top-ranked entity with the ground-truth target."""
        t0 = time.perf_counter()
        question = rec["question"]
        images = rec.get("images")
        sid = rec.get("scene_id")
        gt_entity = rec.get("gt_entity")
        out: Dict[str, Any] = {"scene_id": sid, "condition": condition,
                               "task": "localize", "used_geover": used_geover,
                               "question": question, "expected_answer": rec.get("gt_answer"),
                               "gt_entity": gt_entity, "localized": [], "errors": [],
                               "provenance": {}, "verification": {}, "timings": {}}
        if not Condition.uses_program(condition):
            # without the 3D scene there is nothing to rank; the model is asked to name the
            # target and its answer is scored as a 1-of-N guess over the option set
            res = self.pipeline.run(question=question, options=rec.get("options") or {},
                                    images=images, condition=condition,
                                    used_geover=False, scene_id=sid,
                                    expected_answer=rec.get("gt_answer"))
            d = res.to_dict()
            opts = rec.get("options") or {}
            out["localized"] = [{"entity": opts.get(res.final_answer, ""),
                                 "score": 1.0}] if res.final_answer else []
            out["condition"] = condition
            out["errors"] = d["errors"]
            out["correct"] = d["correct"]
            return out

        from prismv.geover.hypothesis import detect_reference_view

        scene = _materialise_scene(self.pipeline, rec)
        names = [e.name for e in scene.entities.entities] if scene.entities else []
        relation = _relation_from_text(question, names)
        subject = _subject_from_text(question, names)
        ref = detect_reference_view(question)
        ranked: List[Dict] = []
        if relation and subject and names:
            ranked = PRISM_V.rank_relations(scene, anchor=subject,
                                             candidates=[n for n in names if n != subject],
                                             relation=relation, ref_view=ref)
        for cand in ranked:
            err = _position_error(cand, gt_entity, scene)
            cand["position_error"] = err[0]
            cand["scene_extent"] = err[1]
        out["localized"] = ranked
        out["relation"] = relation
        out["subject"] = subject
        out["anchor"] = subject
        out["reference_view"] = ref
        out["provenance"] = dict(getattr(scene, "entity_provenance", {}))
        gt_names = {rec.get("gt_entity"), rec.get("gt_answer")}
        if gt_names and ranked:
            top = ranked[0].get("candidate")
            out["correct"] = (top in gt_names) if top else None
        else:
            out["correct"] = None
        if scene.entities is not None and used_geover:
            out["verification"] = _verify_for_localization(scene, question,
                                                          rec.get("options") or {})
        out["timings"]["localize"] = time.perf_counter() - t0
        out["task"] = "localize"
        return out

    def _run_description(self, rec: Dict, condition: str, used_geover: bool) -> Dict:
        """Scene description: ask for free-form layout text, then check every spatial claim
        in it against the reconstruction.

        Without GeoVer the description is only ever scored by string overlap with a
        reference, which rewards paraphrase; with it, the same text is scored on whether the
        geometry actually supports what it asserts. Those are different properties and the
        two arms must not be compared as if they measured one thing.
        """
        t0 = time.perf_counter()
        question = rec["question"]
        scene = _materialise_scene(self.pipeline, rec) if Condition.uses_program(condition) else None
        prompt = _description_prompt(rec)
        ans = self_pipeline_mllm(self.pipeline, prompt, rec.get("options") or {},
                                 images=rec.get("images") if not scene else None,
                                 ground_with_3d=bool(scene))
        out: Dict[str, Any] = {
            "scene_id": rec.get("scene_id"), "condition": condition, "task": "describe",
            "used_geover": used_geover, "question": question, "gt_relations":
                rec.get("gt_relations"), "description": ans.reasoning or ans.answer,
            "errors": [ans.error] if ans.error else [], "provenance": {},
            "verification": {}, "correct": None, "timings": {},
        }
        if scene is not None:
            out["provenance"] = dict(getattr(scene, "entity_provenance", {}))
            if used_geover:
                from prismv.geover import GeoVer

                gv = GeoVer(graph=scene.entities)
                # the prompt fixes the viewpoint ("from the first image"); the description
                # itself rarely repeats it, so it must be threaded through explicitly
                from prismv.geover.hypothesis import detect_reference_view

                parsed = gv.parse(prompt, {}, reasoning=out["description"],
                                  ref_view=detect_reference_view(prompt))
                report = gv.verify(parsed, [])
                out["verification"] = {"counts": report.counts(),
                                       "checks": [c.to_dict() for c in report.checks],
                                       "n_claims": len(report.checks)}
        out["timings"]["describe"] = time.perf_counter() - t0
        return out

    # ------------------------------------------------------------------------- the sweep
    def run(self, verbose: bool = True) -> List[Cell]:
        self.cells = []
        for task, condition, used_geover in itertools.product(
                self.tasks, self.conditions, self.verification_states):
            cell = Cell(task=task, condition=condition, used_geover=used_geover)
            t0 = time.perf_counter()
            for i, rec in enumerate(self.records, 1):
                try:
                    res = self._run_record(rec, task, condition, used_geover)
                    if not isinstance(res, dict):
                        # a task driver that forgets to serialise would otherwise contribute
                        # zero rows and the cell would report accuracy 0.0 as if it ran
                        raise TypeError(f"{task} driver returned "
                                        f"{type(res).__name__}, expected dict")
                    res.setdefault("task", task)
                    cell.results.append(res)
                except Exception as e:
                    cell.results.append({"scene_id": rec.get("scene_id"), "task": task,
                                         "condition": condition, "used_geover": used_geover,
                                         "errors": [f"runner: {type(e).__name__}: {e}"],
                                         "correct": None})
                if verbose and self.progress_every and i % self.progress_every == 0:
                    print(f"  [{cell.key}] {i}/{len(self.records)}")
            cell.wallclock_s = time.perf_counter() - t0
            self.cells.append(cell)
            if verbose:
                s = cell.summary()
                print(f"[{cell.key}] n={s['n']} acc="
                      f"{s['answer'].get('accuracy', 0.0):.4f} "
                      f"parse={s['health'].get('parse_rate', 0):.2f} "
                      f"exec={s['health'].get('execution_rate', 0):.2f} "
                      f"({cell.wallclock_s:.1f}s)")
        return self.cells

    # ------------------------------------------------------------------------- reporting
    def report(self, path: Optional[str] = None, include_results: bool = True) -> Dict:
        summary: Dict[str, Any] = {
            "n_records": len(self.records),
            "dataset_coverage": task_coverage(self.records),
            "cells": [],
        }
        for cell in self.cells:
            entry = cell.summary()
            entry["condition_comparison"] = metrics.condition_comparison(cell.results)
            entry["localization"] = metrics.localization_metrics(cell.results)
            entry["description"] = metrics.description_metrics(cell.results)
            if include_results:
                entry["results"] = cell.results
            summary["cells"].append(entry)

        # the headline table: same questions, one factor varied at a time
        qa_cells = {f"{c.condition}|{c.used_geover}": c for c in self.cells if c.task == "qa"}
        headline: Dict[str, Any] = {}
        for cond in self.conditions:
            off, on = qa_cells.get(f"{cond}|False"), qa_cells.get(f"{cond}|True")
            if off and on:
                a = [1.0 if r.get("correct") else 0.0 for r in on.results if r.get("correct") is not None]
                b = [1.0 if r.get("correct") else 0.0 for r in off.results if r.get("correct") is not None]
                headline[f"geover_effect_on_{cond}"] = metrics.paired_diff(a, b)
        for cond, other in (("scene_3d", "image_2d"), ("scene_3d", "text_only")):
            c, o = qa_cells.get(f"{cond}|True"), qa_cells.get(f"{other}|True")
            if c and o:
                ra = {r["scene_id"]: (1.0 if r.get("correct") else 0.0)
                      for r in c.results if r.get("correct") is not None and r.get("scene_id")}
                rb = {r["scene_id"]: (1.0 if r.get("correct") else 0.0)
                      for r in o.results if r.get("correct") is not None and r.get("scene_id")}
                common = sorted(set(ra) & set(rb))
                headline[f"{cond}_vs_{other}"] = {
                    "matched_n": len(common),
                    # a comparison over different question subsets must be visibly flagged
                    "matched": len(common) == max(len(ra), len(rb)) and len(common) > 0,
                    **metrics.paired_diff([ra[s] for s in common], [rb[s] for s in common]),
                }
        summary["headline"] = headline
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nEvaluation report -> {path}")
        return summary


# ------------------------------------------------------------------------- task helpers
def _materialise_scene(pipeline: PRISMVPipeline, rec: Dict):
    """Perceive the scene once so task drivers can query it directly."""
    from prismv.pipeline import ToolRecorder

    scene = _make_scene(rec)
    rec_order = getattr(pipeline, "_scene_cache", None)
    if rec_order is None:
        pipeline._scene_cache = {}  # type: ignore[attr-defined]
    cache = pipeline._scene_cache  # type: ignore[attr-defined]
    key = rec.get("scene_id")
    if key and key in cache:
        return cache[key]
    recorder = ToolRecorder()
    code = pipeline._generate_program(scene)
    try:
        from prismv.pipeline import execute_program

        execute_program(code, scene, recorder, 0.0)
    except Exception as e:
        scene.entity_provenance = {"grounding_path": "none", "error": str(e)}
    if scene.reconstruction is None and scene.images:
        try:
            PRISM_V.reconstruct(scene)
        except Exception:
            pass
    if scene.entities is None:
        try:
            PRISM_V.ground_entities(scene, entity_masks=rec.get("entity_masks"),
                                    conf_thres=pipeline.conf_thres,
                                    fallback_clusters=pipeline.fallback_clusters)
        except Exception as e:
            scene.entity_provenance = {"grounding_path": "none", "error": str(e)}
    if key:
        cache[key] = scene
    return scene


def _make_scene(rec: Dict):
    from prismv.interface import Scene

    return Scene(list(rec.get("images") or []), rec.get("question", ""),
                 scene_id=rec.get("scene_id"))


def self_pipeline_mllm(pipeline: PRISMVPipeline, prompt: str, options: Dict[str, str],
                       images: Optional[Sequence[str]] = None,
                       ground_with_3d: bool = False):
    """Single MLLM call through the injected client (used by describe/localize arms)."""
    return pipeline.mllm.answer_question(prompt, options, images=images,
                                         ground_with_3d=ground_with_3d)


def _relation_from_text(text: str, names: Sequence[str]) -> Optional[str]:
    from prismv.geover.predicates import RELATION_ALIASES

    low = " ".join(str(text).lower().split())
    best = None
    for rel, aliases in RELATION_ALIASES.items():
        for a in aliases:
            n = " ".join(a.lower().split())
            if n in low:
                # prefer the longest matching phrasing, so "to the right of" beats "right of"
                if best is None or len(n) > len(best[1]):
                    best = (rel, n)
    return best[0] if best else None


def _subject_from_text(text: str, names: Sequence[str]) -> Optional[str]:
    low = str(text).lower()
    hits = [n for n in names if n and n.lower() in low]
    if not hits:
        return None
    return max(hits, key=len)


def _position_error(cand: Dict, gt_entity: Optional[str], scene) -> tuple:
    """Metres between the predicted entity's centroid and the target's, plus scene extent."""
    graph = getattr(scene, "entities", None)
    if graph is None or not gt_entity:
        return (None, graph.extent() if graph is not None else None)
    from prismv.geover.evidence import GeometricEvidenceIndex

    idx = GeometricEvidenceIndex(graph)
    p = idx.find(cand.get("candidate") or "")
    g = idx.find(gt_entity)
    if p is None or g is None:
        return (None, graph.extent())
    import numpy as np

    return (float(np.linalg.norm(p.centroid - g.centroid)), graph.extent())


def _verify_for_localization(scene, question: str, options: Dict[str, str]) -> Dict:
    from prismv.geover import GeoVer

    gv = GeoVer(graph=scene.entities)
    parsed = gv.parse(question, options)
    report = gv.verify(parsed, list(options.keys()))
    return {"counts": report.counts(), "option_coverage": report.option_coverage(),
            "checks": [c.to_dict() for c in report.checks]}


def _description_prompt(rec: Dict) -> str:
    """Prompt for the scene-description task.

    Kept as an explicit function with the entity list embedded, because "describe the scene"
    with no vocabulary anchor produces free text that the checker cannot bind to any entity,
    and the resulting low claim count would look like a verification failure rather than a
    prompt failure.
    """
    names = rec.get("entity_names") or []
    base = ("Describe the spatial layout of this scene: for each pair of objects you can "
            "identify, state where one is relative to the other (left/right, nearer/farther, "
            "above/below) from the viewpoint of the first image.")
    if names:
        base += " Use only these object names: " + ", ".join(str(n) for n in names) + "."
    q = rec.get("question")
    return f"{base}\n\nContext: {q}" if q else base
