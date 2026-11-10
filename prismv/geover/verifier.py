"""Constraint checking for GeoVer (PRISM-V).

Given retrieved geometric evidence and a parsed spatial hypothesis, decide whether each
assertion is supported, contradicted, or simply undetermined by the geometry -- and keep
the evidence trail that decided it.

The three-way outcome is not cosmetic. A binary verifier that reports "not supported"
for both *contradicted* and *cannot tell* will happily demote a correct answer just
because the reconstruction was sparse; treating those apart is what makes retrieval-based
verification additive rather than a second opinion that can override perception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


from prismv.geover.evidence import GeometricEvidenceIndex
from prismv.geover.hypothesis import Assertion
from prismv.geover.predicates import MIN_POINTS, PREDICATES, Verdict


#: keys carrying measurement overhead rather than geometric judgement. The audit trail must
#: be a pure function of the scene, so reproducibility checks strip exactly these.
INSTRUMENT_KEYS = ("cost_us", "predicate_ms")


@dataclass
class Check:
    """One assertion after it has been run against the geometry."""

    assertion: Assertion
    verdict: Verdict
    status: str                     # "supported" | "contradicted" | "undetermined"
    entities_used: List[str] = field(default_factory=list)
    cost_us: int = 0

    def to_dict(self) -> Dict:
        a = self.assertion
        return {
            "relation": a.relation,
            "subject": a.subject,
            "object": a.object,
            "third": a.third,
            "ref_view": a.ref_view,
            "source": a.source,
            "status": self.status,
            "score": round(self.verdict.score, 4),
            "margin": round(self.verdict.margin, 4),
            "detail": self.verdict.detail,
            "entities_used": self.entities_used,
            # Instrumentation, not verdict content. Without a cost figure "lightweight
            # verification layer" is just an adjective; but timing must never leak into the
            # reproducibility guarantee, so these keys are listed in INSTRUMENT_KEYS and
            # stripped by the determinism test rather than being omitted outright.
            "cost_us": self.cost_us,
            "predicate_ms": round(self.cost_us / 1000.0, 4),
        }


@dataclass
class VerificationReport:
    """All checks for one question, grouped the way the reranker needs them."""

    checks: List[Check] = field(default_factory=list)
    per_option: Dict[str, List[Check]] = field(default_factory=dict)
    index_stats: Dict = field(default_factory=dict)
    skipped_reason: str = ""

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    def counts(self) -> Dict[str, int]:
        c = {"supported": 0, "contradicted": 0, "undetermined": 0}
        for ch in self.checks:
            c[ch.status] += 1
        return c

    def option_coverage(self) -> Dict[str, Dict[str, int]]:
        """Per-option support counts: the diagnostic that explains a rerank, and the
        table the ablation sweep reports GeoVer's actual reach against."""
        out: Dict[str, Dict[str, int]] = {}
        for letter, chs in self.per_option.items():
            d = {"supported": 0, "contradicted": 0, "undetermined": 0}
            for ch in chs:
                d[ch.status] += 1
            out[letter] = d
        return out

    def to_dict(self) -> Dict:
        return {
            "counts": self.counts(),
            "option_coverage": self.option_coverage(),
            "index_stats": self.index_stats,
            "skipped_reason": self.skipped_reason,
            "checks": [c.to_dict() for c in self.checks],
        }


def classify(assertion: Assertion, verdict: Verdict) -> Optional[str]:
    """Map a graded verdict onto a three-way constraint status.

    ``near``/``far_from`` are threshold-defined, so they only count as hard constraints
    when the margin is decisive; a borderline near/far is evidence, not a contradiction.
    """
    if verdict.undetermined:
        return None
    if assertion.relation in ("near", "far_from"):
        if abs(verdict.margin) < 0.2:
            return "undetermined"
    return "supported" if verdict.holds else "contradicted"


class GeoVerifier:
    """Runs parsed assertions against the retrieved geometric evidence."""

    def __init__(self, index: GeometricEvidenceIndex, *,
                 undetermined_score: float = 0.5,
                 min_entity_points: int = MIN_POINTS,
                 enabled_relations: Optional[Sequence[str]] = None):
        self.index = index
        self.undetermined_score = float(undetermined_score)
        self.min_entity_points = int(min_entity_points)
        self.enabled = set(enabled_relations) if enabled_relations else set(PREDICATES)

    # ------------------------------------------------------------------ single check
    def check(self, assertion: Assertion) -> Optional[Check]:
        t0 = time.perf_counter()
        if assertion.relation not in self.enabled:
            return None
        if not assertion.is_complete():
            return None
        fn = PREDICATES.get(assertion.relation)
        if fn is None:
            return None
        bundle = self.index.bundle_for(assertion)
        if not bundle.sufficient:
            return Check(assertion, Verdict(False, self.undetermined_score, 0.0, True,
                                            bundle.reason), "undetermined", [])
        # evidence-density gate: a 4-pixel entity is a reconstruction artefact, not a fact
        for slot in ("subject", "object", "third"):
            e = bundle.entities.get(slot)
            if e is not None and e.n_points < self.min_entity_points:
                return Check(assertion,
                             Verdict(False, self.undetermined_score, 0.0, True,
                                     f"{slot} '{e.name}' has {e.n_points} points "
                                     f"(<{self.min_entity_points})"),
                             "undetermined", [e.name])
        try:
            verdict = fn(bundle.entities["subject"], bundle.entities["object"], bundle)
        except Exception as exc:  # a broken predicate must never take down the answer
            verdict = Verdict(False, self.undetermined_score, 0.0, True, f"error: {exc}")
        status = classify(assertion, verdict) or "undetermined"
        used = [e.name for e in bundle.entities.values() if e is not None]
        cost = int((time.perf_counter() - t0) * 1e6)
        return Check(assertion, verdict, status, used, cost)

    # ---------------------------------------------------------------- batch per option
    def verify(self, parsed, option_letters: Optional[Sequence[str]] = None) -> VerificationReport:
        """Verify question/reasoning/description assertions plus per-option assertions."""
        report = VerificationReport(index_stats=self.index.stats())
        pools: List[Assertion] = []
        pools += list(parsed.question_assertions) if hasattr(parsed, "question_assertions") else []
        pools += list(getattr(parsed, "reasoning_assertions", []))
        opt_assertions = list(getattr(parsed, "option_assertions", []))
        letters = list(option_letters) if option_letters else sorted(
            {a.source.split(":")[-1] for a in opt_assertions})
        for letter in letters:
            report.per_option[letter] = []

        for a in pools:
            ch = self.check(a)
            if ch:
                report.checks.append(ch)
        for a in opt_assertions:
            ch = self.check(a)
            if not ch:
                continue
            report.checks.append(ch)
            letter = a.source.split(":")[-1]
            report.per_option.setdefault(letter, []).append(ch)
        if not report.checks:
            report.skipped_reason = ("no checkable assertion (extraction found no "
                                     "grounded entity pair)")
        return report
