"""GeoVer: retrieval-based geometric verification for PRISM-V.

Public entry point. The intended call sequence mirrors the paper's pipeline
(perceive -> verify -> reason):

    gv = GeoVer(processed_dir=..., reconstruction=scene.reconstruction, ...)
    out = gv.verify_and_rerank(question=..., options=..., vlm_answer="B",
                               vlm_confidence=[...], reasoning=...)

Every stage is reachable independently so the analysis sweep can ablate verification
without touching perception, and ablate grounding without touching the reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from prismv.geover.evidence import (EntityEvidence, EvidenceBundle,
                                    GeometricEvidenceIndex, SceneGraph)
from prismv.geover.grounding import build_scene_graph, cluster_entities, lift_masks, load_pointmap
from prismv.geover.hypothesis import (
    Assertion,
    ParsedHypotheses,
    assertions_from_text,
    assertions_for_options,
    detect_reference_view,
    extract_clauses,
    match_entities,
)
from prismv.geover.predicates import (
    MIN_POINTS,
    PREDICATES,
    RELATION_ALIASES,
    VIEW_RELATIVE,
    Verdict,
    estimate_gravity,
    infer_pose_convention,
    relation_support_report,
)
from prismv.geover.rerank import AnswerReranker, OptionScore, RerankResult
from prismv.geover.verifier import Check, GeoVerifier, VerificationReport


@dataclass
class GeoVerOutput:
    """Everything one verification pass produces, including the audit trail."""

    answer: str
    applied: bool
    report: VerificationReport
    rerank: RerankResult
    parsed: ParsedHypotheses
    graph: SceneGraph
    provenance: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "applied": self.applied,
            "provenance": self.provenance,
            "hypotheses": parsed_summary(self.parsed),
            "verification": self.report.to_dict(),
            "rerank": self.rerank.to_dict(),
            "scene_graph": {
                "entities": [e.to_dict() for e in self.graph.entities],
                **self.report.index_stats,
            },
        }


def parsed_summary(parsed: ParsedHypotheses) -> Dict:
    return {
        **parsed.summary(),
        "assertions": [a.to_dict() for a in
                       list(parsed.question_assertions) + list(parsed.option_assertions) +
                       list(parsed.reasoning_assertions)],
    }


class GeoVer:
    """Retrieval-based geometric verification layer.

    Parameters
    ----------
    processed_dir :
        Directory holding ``pointmaps.npz`` / ``entities.json`` written by the PRISM-V
        reconstruction pipeline. Optional; without it verification runs on whatever is
        passed via ``graph`` / ``reconstruction`` and usually abstains.
    graph :
        A prebuilt :class:`SceneGraph`. Takes precedence over ``processed_dir``.
    reconstruction :
        A :class:`Reconstruction`-like object (point_cloud / extrinsics / intrinsics).
    verification_weight :
        Lambda on the verification term in the reranking objective. 0 disables reranking
        while still producing the report -- this is the ``-GeoVer`` ablation arm.
    min_support :
        How many hard-supported constraints an option needs before the layer is allowed to
        override the MLLM. The single most important robustness knob.
    enabled_relations :
        Restrict the predicate vocabulary (used by the relation-coverage analysis).
    """

    def __init__(self, processed_dir: Optional[str] = None,
                 graph: Optional[SceneGraph] = None,
                 reconstruction=None,
                 entity_masks: Optional[Sequence[Dict]] = None,
                 conf_thres: float = 0.0,
                 fallback_clusters: int = 0,
                 verification_weight: float = 1.0,
                 min_support: int = 1,
                 contradiction_penalty: float = 1.0,
                 min_entity_points: int = MIN_POINTS,
                 enabled_relations: Optional[Sequence[str]] = None,
                 scene_id: Optional[str] = None):
        if graph is not None:
            self.graph, self.provenance = graph, {"grounding_path": "provided"}
        else:
            self.graph, self.provenance = build_scene_graph(
                processed_dir=processed_dir, reconstruction=reconstruction,
                entity_masks=entity_masks, conf_thres=conf_thres,
                fallback_clusters=fallback_clusters, scene_id=scene_id)
        self.index = GeometricEvidenceIndex(self.graph)
        self.verifier = GeoVerifier(self.index, min_entity_points=min_entity_points,
                                   enabled_relations=enabled_relations)
        self.reranker = AnswerReranker(weight=verification_weight, min_support=min_support,
                                       contradiction_penalty=contradiction_penalty)
        self.enabled = bool(self.graph.entities)

    # ---------------------------------------------------------------------- helpers
    def parse(self, question: str, options: Optional[Dict[str, str]] = None,
              reasoning: str = "", ref_view: Optional[int] = None) -> ParsedHypotheses:
        """Compile a question (and optional reasoning/description text) into assertions.

        ``ref_view`` pins the viewer for claims that carry no viewpoint of their own. It
        matters for the description task in particular: a generated description states "the
        cup is to the right of the book" without repeating which image the layout was
        described from, so the view has to come from the task prompt. Left unset, every
        lateral claim from such text is correctly reported as undetermined -- but then the
        checker never actually exercises its main capability on that task.
        """
        parsed = assertions_for_options(question, options or {}, self.index)
        if ref_view is not None:
            parsed.ref_view = ref_view
            for a in list(parsed.question_assertions) + list(parsed.option_assertions):
                if a.ref_view is None:
                    a.ref_view = ref_view
        if reasoning:
            parsed.reasoning_assertions = assertions_from_text(reasoning, self.index,
                                                              source="reasoning",
                                                              ref_view=ref_view)
        return parsed

    def verify(self, parsed: ParsedHypotheses,
               letters: Optional[Sequence[str]] = None) -> VerificationReport:
        return self.verifier.verify(parsed, letters)

    # -------------------------------------------------------------------- main entry
    def verify_and_rerank(self, question: str, options: Dict[str, str],
                          vlm_answer: Optional[str] = None,
                          vlm_confidence: Optional[Sequence[float]] = None,
                          reasoning: str = "") -> GeoVerOutput:
        """Verify the model's spatial hypotheses and rerank the candidate answers."""
        letters = list(options.keys())
        if not self.enabled:
            empty = VerificationReport(skipped_reason="no grounded entities in scene")
            ranked = AnswerReranker(weight=0.0).score(empty, letters, vlm_confidence,
                                                       vlm_answer)
            return GeoVerOutput(answer=vlm_answer or ranked.answer, applied=False,
                                report=empty, rerank=ranked,
                                parsed=ParsedHypotheses(), graph=self.graph,
                                provenance={**self.provenance, "reason": "no entities"})
        parsed = self.parse(question, options, reasoning)
        report = self.verify(parsed, letters)
        rerank = self.reranker.score(report, letters, vlm_confidence, vlm_answer)
        return GeoVerOutput(answer=rerank.answer, applied=rerank.applied, report=report,
                            rerank=rerank, parsed=parsed, graph=self.graph,
                            provenance=self.provenance)

    # ------------------------------------------------------------------- introspection
    def stats(self) -> Dict:
        return {"index": self.index.stats(), "grounding": self.provenance,
                "enabled_relations": sorted(self.verifier.enabled),
                "view_relative_relations": sorted(VIEW_RELATIVE)}


__all__ = ["GeoVer", "GeoVerOutput", "SceneGraph", "EntityEvidence", "GeometricEvidenceIndex",
           "GeoVerifier", "VerificationReport", "Check", "AnswerReranker", "RerankResult",
           "OptionScore", "Assertion", "ParsedHypotheses", "EvidenceBundle", "Verdict",
           "build_scene_graph", "lift_masks", "cluster_entities", "load_pointmap",
           "assertions_for_options", "assertions_from_text", "extract_clauses",
           "detect_reference_view", "match_entities", "estimate_gravity",
           "infer_pose_convention", "relation_support_report", "PREDICATES",
           "RELATION_ALIASES", "VIEW_RELATIVE", "MIN_POINTS"]
