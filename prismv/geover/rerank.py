"""Answer reranking for GeoVer (PRISM-V).

The reranker turns verification into a decision. Its design constraint is that
verification must be able to *refuse*: geometric evidence is only trustworthy when the
scene was reconstructed well and the entities were grounded, so an absent or borderline
constraint should leave the MLLM's original ranking untouched.

That is what the ``min_support`` gate and the abstention flag do. Skipping them turns
GeoVer into a veto that fires on noise, which is the usual way a verification layer ends
up scoring below the baseline it verifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from prismv.geover.verifier import VerificationReport


@dataclass
class OptionScore:
    """Final per-option accounting, kept explicit enough to audit a rerank afterwards."""

    letter: str
    prior: float                       # MLLM belief in this option, normalised
    support: int = 0
    contradict: int = 0
    undetermined: int = 0
    mean_score: float = 0.5            # average graded geometric support
    verification: float = 0.0          # signed aggregate evidence, in [-1, 1]
    combined: float = 0.0
    changed: bool = False
    abstained: bool = False
    detail: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"option": self.letter, "prior": round(self.prior, 4),
                "support": self.support, "contradict": self.contradict,
                "undetermined": self.undetermined, "mean_score": round(self.mean_score, 4),
                "verification": round(self.verification, 4),
                "combined": round(self.combined, 6), "changed": self.changed,
                "abstained": self.abstained}


@dataclass
class RerankResult:
    answer: str
    ranked: List[OptionScore]
    applied: bool                      # False when verification abstained
    margin: float = 0.0                # combined gap between top-1 and top-2
    note: str = ""

    def to_dict(self) -> Dict:
        return {"answer": self.answer, "applied": self.applied,
                "margin": round(self.margin, 4), "note": self.note,
                "ranked": [r.to_dict() for r in self.ranked]}


def _priors_from_confidence(confidence: Optional[Sequence[float]],
                            letters: Sequence[str]) -> Dict[str, float]:
    """Normalise per-option logprobs into a belief distribution.

    When the model exposes option logprobs, the prior is genuinely its belief and the
    rerank is a proper posterior; when it does not, the prior is flat and only the
    verification term moves the answer. Both regimes are reported so the ablation can
    separate "rerank helped" from "prior was already right".
    """
    letters = list(letters)
    if confidence is None or len(confidence) != len(letters):
        return {l: 1.0 / max(1, len(letters)) for l in letters}
    c = np.asarray(confidence, dtype=np.float64)
    c = np.nan_to_num(c, nan=-100.0, posinf=0.0, neginf=-100.0)
    if np.all(c <= -1e6):
        return {l: 1.0 / max(1, len(letters)) for l in letters}
    c = c - c.max()
    w = np.exp(c)
    s = w.sum()
    if s <= 1e-12:
        return {l: 1.0 / max(1, len(letters)) for l in letters}
    w = w / s
    return {l: float(w[i]) for i, l in enumerate(letters)}


class AnswerReranker:
    """Combine MLLM answer priors with geometric verification into a final ranking."""

    def __init__(self, weight: float = 1.0, min_support: int = 1,
                 contradiction_penalty: float = 1.0, abstain_threshold: float = 0.02,
                 undetermined_score: float = 0.5):
        self.weight = float(weight)                    # lambda in the paper's objective
        self.min_support = int(min_support)            # evidence needed before overriding
        self.contradiction_penalty = float(contradiction_penalty)
        self.abstain_threshold = float(abstain_threshold)
        self.undetermined_score = float(undetermined_score)

    def score(self, report: VerificationReport, letters: Sequence[str],
              prior: Optional[Sequence[float]] = None,
              original_answer: Optional[str] = None,
              coverage: Optional[Dict[str, float]] = None) -> RerankResult:
        """Rank options from verification evidence.

        ``coverage`` optionally supplies a per-option fraction in [0, 1] of how much of the
        option's asserted content could be checked at all. Weighting the evidence term by
        it stops an option from winning merely because it mentioned fewer entities.
        """
        letters = list(letters)
        if not letters:
            return RerankResult(answer=original_answer or "", ranked=[], applied=False,
                                note="no options")
        scores = _priors_from_confidence(prior, letters)
        rows: List[OptionScore] = []
        for letter in letters:
            chs = report.per_option.get(letter, [])
            sup = sum(1 for c in chs if c.status == "supported")
            con = sum(1 for c in chs if c.status == "contradicted")
            und = sum(1 for c in chs if c.status == "undetermined")
            graded = [c.verdict.score for c in chs if not c.verdict.undetermined]
            mean_score = float(np.mean(graded)) if graded else self.undetermined_score
            # signed aggregate: how much the retrievable geometry endorses this option
            if sup + con > 0:
                verification = (sup - self.contradiction_penalty * con) / (sup + con)
            else:
                verification = 0.0
            if coverage is not None and letter in coverage:
                verification *= float(np.clip(coverage[letter], 0.0, 1.0))
            base = scores[letter]
            combined = base + self.weight * verification * base
            rows.append(OptionScore(letter=letter, prior=scores[letter], support=sup,
                                   contradict=con, undetermined=und,
                                   mean_score=mean_score, verification=verification,
                                   combined=combined,
                                   detail=[c.to_dict() for c in chs]))

        ranked = sorted(rows, key=lambda r: -r.combined)
        total_evidence = sum(r.support + r.contradict for r in ranked)
        top = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        margin = float(top.combined - runner.combined) if runner else 1.0

        # abstain when there is essentially no hard geometric evidence, or when the
        # surviving gap is smaller than the noise in the evidence itself
        decisive = top.support >= self.min_support or any(r.contradict for r in ranked)
        applied = bool(decisive and total_evidence > 0 and margin > self.abstain_threshold)
        if not applied:
            for r in ranked:
                r.abstained = True

        new_answer = top.letter if applied else (original_answer or ranked[0].letter)
        changed = bool(applied and original_answer and new_answer != original_answer)
        for r in ranked:
            r.changed = changed if r.letter == new_answer else False
        note = ("applied" if applied else
                ("abstained: no hard geometric evidence" if total_evidence == 0 else
                 "abstained: below min_support / margin threshold"))
        return RerankResult(answer=new_answer, ranked=ranked, applied=applied,
                            margin=margin, note=note)
