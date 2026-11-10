"""Evaluation metrics for PRISM-V's three tasks.

The metrics are written so that the verification layer can be judged *net* rather than on
the cases it flatters. Two numbers do most of that work:

``fix_rate``     questions the answer went wrong -> right because of GeoVer.
``damage_rate``  questions the answer went right -> wrong because of GeoVer.

A verification layer whose damage_rate is not reported is not measurable, and a paper that
reports only accuracy deltas cannot tell you which of the two produced the delta.

Everything here is pure-python/numpy and consumes serialised result dicts, so metrics can
be recomputed on a finished run without re-running the model.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------- helpers
def _letter(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    return s[0] if s and s[0].isalpha() else s


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI over per-question scores.

    Spatial benchmarks in this family are small (tens to a few hundred items), so the
    difference between two conditions is often inside the noise of a point estimate.
    Reporting the interval is what makes "PRISM-V improves over the 2D condition" a claim
    that can be checked rather than admired.
    """
    v = np.asarray([x for x in values if x == x], dtype=np.float64)
    if len(v) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return {"mean": round(float(v.mean()), 4),
            "lo": round(float(np.percentile(means, 100 * alpha / 2)), 4),
            "hi": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4),
            "n": int(len(v))}


def paired_diff(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0,
                alpha: float = 0.05) -> Dict[str, float]:
    """Paired bootstrap on A - B over matched questions.

    Paired, not independent: the two conditions see the same scenes, so unpaired intervals
    throw away the correlation and understate what is actually known.
    """
    x = np.asarray([v for v in a], dtype=np.float64)
    y = np.asarray([v for v in b], dtype=np.float64)
    k = min(len(x), len(y))
    if k == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "p_gt_zero": float("nan")}
    d = x[:k] - y[:k]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, k, size=(n, k))
    means = d[idx].mean(axis=1)
    return {"delta": round(float(d.mean()), 4),
            "lo": round(float(np.percentile(means, 100 * alpha / 2)), 4),
            "hi": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4),
            "n": int(k),
            "p_gt_zero": round(float((means <= 0).mean()), 4)}


# --------------------------------------------------------------------- task 1: 3D QA
def answer_accuracy(results: Iterable[Dict], group_by: Optional[str] = None) -> Dict:
    """Answer correctness overall, optionally split by an attribute of each result.

    ``group_by`` accepts ``"condition"``, ``"used_geover"``, ``"scene_type"``,
    ``"relation"`` or any key present in the result dicts. Only evaluable items count, and
    the evaluable/total split is reported so a low count is visible rather than hidden
    inside a flattering percentage.
    """
    buckets: Dict[Any, List[float]] = defaultdict(list)
    all_scores: List[float] = []
    evaluable = skipped = 0
    for r in results:
        ok = r.get("correct")
        if ok is None:
            skipped += 1
            continue
        evaluable += 1
        score = 1.0 if ok else 0.0
        all_scores.append(score)
        key = r.get(group_by) if group_by else "all"
        if isinstance(key, (list, set)):
            key = tuple(sorted(key))
        buckets[key].append(score)

    out = {
        "n_evaluable": evaluable,
        "n_skipped_no_gt": skipped,
        "accuracy": round(float(np.mean(all_scores)), 4) if all_scores else 0.0,
    }
    out.update(bootstrap_ci(all_scores))
    if group_by:
        out["by_group"] = {str(k): {"n": len(v), "accuracy": round(float(np.mean(v)), 4),
                                   **bootstrap_ci(v)}
                           for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}
    return out


def verification_effect(results: Iterable[Dict]) -> Dict:
    """Net effect of the verification layer, decomposed into fixes and damages.

    Split by whether GeoVer actually applied a rerank: ``applied=False`` is the layer
    abstaining, which should leave the answer untouched and therefore contribute to neither
    the fix nor the damage count.
    """
    fix = damage = neutral = 0
    applied_n = abstained_n = 0
    uniquely_endorsed = ambiguous = n_with_coverage = 0
    flip_to_correct: List[float] = []
    rows = []
    for r in results:
        if not r.get("used_geover"):
            continue
        ver = r.get("verification") or {}
        gt = _letter(r.get("expected_answer"))
        pre = _letter(r.get("mllm_answer"))
        post = _letter(r.get("final_answer"))
        if not gt or not pre or not post:
            continue
        # How much of the decision the constraint set could actually make. Counting options
        # that the geometry endorses and does not contradict tells you the ceiling of
        # reranking: with more than one endorsed option the layer has no basis to choose
        # between them, and any accuracy change measured on those items is prior noise, not
        # verification.
        cov = ver.get("option_coverage") or {}
        if cov:
            endorsed = [l for l, c in cov.items()
                        if c.get("supported", 0) > 0 and c.get("contradicted", 0) == 0]
            if len(endorsed) == 1:
                uniquely_endorsed += 1
            elif len(endorsed) > 1:
                ambiguous += 1
            n_with_coverage += 1
        if ver.get("applied"):
            applied_n += 1
        else:
            abstained_n += 1
            continue
        if pre != gt and post == gt:
            fix += 1
            flip_to_correct.append(1.0)
        elif pre == gt and post != gt:
            damage += 1
            flip_to_correct.append(0.0)
        else:
            neutral += 1
        rows.append({"scene_id": r.get("scene_id"), "gt": gt, "pre": pre, "post": post,
                     "changed": bool(ver.get("changed_answer")),
                     "note": ver.get("note", "")})
    n = applied_n
    return {
        "n_questions": len(flip_to_correct),
        "n_applied_rerank": applied_n,
        "n_abstained": abstained_n,
        "fixes": fix,
        "damages": damage,
        "net": fix - damage,
        "fix_rate": round(fix / n, 4) if n else 0.0,
        "damage_rate": round(damage / n, 4) if n else 0.0,
        "net_effect": round((fix - damage) / n, 4) if n else 0.0,
        "applic_rate": round(applied_n / max(1, applied_n + abstained_n), 4),
        # how often the retrieved geometry actually pins down a single option. This bounds
        # everything above: on the ambiguous remainder the layer can only demote a
        # contradicted answer, never select the right one, so a net effect computed over all
        # questions silently averages two very different regimes.
        "n_with_coverage": n_with_coverage,
        "n_uniquely_endorsed": uniquely_endorsed,
        "n_ambiguous": ambiguous,
        "constraint_set_decisive_rate": round(uniquely_endorsed / n_with_coverage, 4)
        if n_with_coverage else 0.0,
        "applied_case_accuracy": round(float(np.mean(flip_to_correct)), 4) if flip_to_correct else 0.0,
        "examples": rows[:20],
    }


def relation_level_accuracy(results: Iterable[Dict]) -> Dict:
    """Accuracy bucketed by the spatial relation each question exercised.

    This is the table that tells you *which* geometry the verification layer actually
    buys; a uniform improvement is far less informative than "left/right from a specified
    view improves 12 points, verticality does not move".
    """
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        gt, post = _letter(r.get("expected_answer")), _letter(r.get("final_answer"))
        if not gt or not post:
            continue
        score = 1.0 if gt == post else 0.0
        rels = set()
        for chk in (r.get("verification") or {}).get("checks", []):
            if chk.get("relation"):
                rels.add(chk["relation"])
        if not rels:
            rels = {"(no verifiable relation)"}
        for rel in rels:
            buckets[rel].append(score)
    return {rel: {"n": len(v), "accuracy": round(float(np.mean(v)), 4)}
            for rel, v in sorted(buckets.items())}


def grounding_quality(results: Iterable[Dict]) -> Dict:
    """How much of each scene actually got grounded -- the upper bound on verification."""
    paths = Counter()
    n_entities: List[float] = []
    unresolved: List[float] = []
    for r in results:
        prov = r.get("provenance") or {}
        paths[prov.get("grounding_path", "none")] += 1
        ents = (r.get("verification") or {}).get("index_stats", {})
        n_entities.append(float(prov.get("n_entities_grounded", 0) or 0))
        cov = (r.get("verification") or {}).get("option_coverage", {})
        tot = sum(sum(c.values()) for c in cov.values())
        und = sum(c.get("undetermined", 0) for c in cov.values())
        unresolved.append(und / tot if tot else 1.0)
    return {
        "grounding_paths": dict(paths),
        "mean_entities_grounded": round(float(np.mean(n_entities)), 3) if n_entities else 0.0,
        "scenes_with_zero_entities": int(sum(1 for v in n_entities if v == 0)),
        "mean_undetermined_share": round(float(np.mean(unresolved)), 4) if unresolved else 0.0,
    }


# --------------------------------------------------------------------- task 2: description
def description_metrics(results: Iterable[Dict]) -> Dict:
    """Score generated scene descriptions against the geometry.

    Two separate things are being measured, and conflating them is how spatial-description
    work hides hallucination:

    ``geometric_precision``  of the claims that were checkable, how many the geometry
                           supports. This is the hallucination proxy.
    ``relation_recall``      of the ground-truth relations, how many the description
                           asserted at all. Low recall with high precision just means the
                           model said very little, confidently.
    """
    tot_checkable = tot_supported = tot_contradicted = 0
    prec_per_item: List[float] = []
    recall_per_item: List[float] = []
    hallucination_rate: List[float] = []
    for r in results:
        checks = (r.get("verification") or {}).get("checks", [])
        if not checks:
            continue
        sup = sum(1 for c in checks if c.get("status") == "supported")
        con = sum(1 for c in checks if c.get("status") == "contradicted")
        und = sum(1 for c in checks if c.get("status") == "undetermined")
        checkable = sup + con
        if checkable:
            prec_per_item.append(sup / checkable)
            hallucination_rate.append(con / checkable)
        tot_checkable += checkable + und
        tot_supported += sup
        tot_contradicted += con
        gt_rels = r.get("gt_relations") or []
        if gt_rels:
            asserted = {c.get("relation") for c in checks}
            hit = sum(1 for g in gt_rels if g in asserted)
            recall_per_item.append(hit / len(gt_rels))
    out = {
        "n_scored": len(prec_per_item),
        "geometric_precision": round(float(np.mean(prec_per_item)), 4) if prec_per_item else 0.0,
        "contradiction_rate": round(float(np.mean(hallucination_rate)), 4) if hallucination_rate else 0.0,
        "total_supported_claims": tot_supported,
        "total_contradicted_claims": tot_contradicted,
        "total_undetermined_claims": tot_checkable - tot_supported - tot_contradicted,
    }
    if recall_per_item:
        out["relation_recall"] = round(float(np.mean(recall_per_item)), 4)
    out.update(bootstrap_ci(prec_per_item) if prec_per_item else {})
    return out


# --------------------------------------------------------------------- task 3: localisation
def localization_metrics(results: Iterable[Dict], thresholds: Sequence[float] = (0.1, 0.25, 0.5),
                         unit: str = "extent") -> Dict:
    """Spatial localisation accuracy: hit rates plus a scale-normalised position error.

    ``unit="extent"`` divides distances by the scene's own median entity separation, which
    is the only way a single threshold means anything across both a bedroom and a warehouse.
    Report raw metres alongside, or the number is untransferable.
    """
    hits: List[float] = []
    hits_k: Dict[int, List[float]] = defaultdict(list)
    errors: List[float] = []
    raw_errors: List[float] = []
    per_threshold: Dict[float, List[float]] = {t: [] for t in thresholds}
    for r in results:
        gt = r.get("gt_entity") or _letter(r.get("expected_answer"))
        preds = r.get("localized") or []
        if not gt or not preds:
            continue
        names = [p.get("entity") or p.get("candidate") for p in preds]
        top = names[0] if names else None
        correct = 1.0 if top and _letter(top) == _letter(gt) else 0.0
        hits.append(correct)
        for k in (1, 3, 5):
            hits_k[k].append(1.0 if _letter(gt) in [_letter(n) for n in names[:k]] else 0.0)
        err = preds[0].get("position_error")
        ext = preds[0].get("scene_extent") or 1.0
        if err is not None:
            raw_errors.append(float(err))
            errors.append(float(err) / max(float(ext), 1e-6))
        for t in thresholds:
            if err is not None:
                per_threshold[t].append(1.0 if float(err) / max(float(ext), 1e-6) <= t else 0.0)
    out = {
        "n_scored": len(hits),
        "localization_accuracy_at_1": round(float(np.mean(hits)), 4) if hits else 0.0,
        "localization_accuracy_at_3": round(float(np.mean(hits_k[3])), 4) if hits_k[3] else 0.0,
        "localization_accuracy_at_5": round(float(np.mean(hits_k[5])), 4) if hits_k[5] else 0.0,
        "unit": unit,
    }
    if errors:
        out["median_position_error_extent_units"] = round(float(np.median(errors)), 4)
        out["mean_position_error_extent_units"] = round(float(np.mean(errors)), 4)
        out["mean_position_error_metres"] = round(float(np.mean(raw_errors)), 4)
    for t in thresholds:
        if per_threshold[t]:
            out[f"accuracy_within_{t}"] = round(float(np.mean(per_threshold[t])), 4)
    return out


# --------------------------------------------------------------------- execution health
def pipeline_health(results: Iterable[Dict]) -> Dict:
    """Stage-by-stage attrition, which is what an end-to-end number cannot show.

    If only 40% of programs execute and accuracy still looks good, the headline number is
    an average over the easy half of the benchmark. Counting parse, execution, grounding and
    answer-generation rates separately is the difference between knowing that and guessing.
    """
    n = len(results)
    if not n:
        return {}
    rs = list(results)
    def rate(pred) -> float:
        return round(sum(1 for r in rs if pred(r)) / n, 4)
    errs = Counter()
    for r in rs:
        for e in r.get("errors", []):
            errs[e.split(":")[0][:48]] += 1
    tools = Counter()
    calls = 0
    for r in rs:
        for t in (r.get("tool_summary") or {}).get("sequence", []):
            tools[t] += 1
            calls += 1
    return {
        "n": n,
        "parse_rate": rate(lambda r: r.get("parse_ok")),
        "execution_rate": rate(lambda r: r.get("execution_ok")),
        "grounded_rate": rate(lambda r: (r.get("provenance") or {}).get("n_entities_grounded", 0)),
        "answered_rate": rate(lambda r: bool(r.get("final_answer"))),
        "evaluable_rate": rate(lambda r: r.get("correct") is not None),
        "mean_tool_calls": round(calls / n, 2),
        "tool_frequency": dict(tools.most_common()),
        "mean_verification_checks": round(
            float(np.mean([len((r.get("verification") or {}).get("checks", [])) for r in rs])), 2),
        "top_errors": dict(errs.most_common(8)),
    }


# --------------------------------------------------------------------- condition comparison
def condition_comparison(results: Iterable[Dict],
                         conditions: Sequence[str] = ("text_only", "image_2d", "scene_3d")
                         ) -> Dict[str, Any]:
    """Matched comparison across input conditions, with paired intervals.

    Matched on ``scene_id``: conditions must be evaluated on the same question set for the
    paired difference to mean anything. Unmatched comparisons are silently reported as
    unpaired and flagged.
    """
    by_condition: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in results:
        c = r.get("condition")
        sid = r.get("scene_id")
        if c is None or sid is None or r.get("correct") is None:
            continue
        by_condition[c][sid] = 1.0 if r["correct"] else 0.0
    table: Dict[str, Any] = {}
    for c in conditions:
        if c in by_condition:
            vals = list(by_condition[c].values())
            table[c] = {"n": len(vals), "accuracy": round(float(np.mean(vals)), 4),
                        **bootstrap_ci(vals)}
    if "scene_3d" in by_condition and "image_2d" in by_condition:
        common = sorted(set(by_condition["scene_3d"]) & set(by_condition["image_2d"]))
        table["scene_3d_vs_image_2d"] = {
            "matched_n": len(common),
            **paired_diff([by_condition["scene_3d"][s] for s in common],
                          [by_condition["image_2d"][s] for s in common]),
            "matched": len(common) == max(len(by_condition["scene_3d"]),
                                          len(by_condition["image_2d"])),
        }
    if "scene_3d" in by_condition and "text_only" in by_condition:
        common = sorted(set(by_condition["scene_3d"]) & set(by_condition["text_only"]))
        table["scene_3d_vs_text_only"] = {
            "matched_n": len(common),
            **paired_diff([by_condition["scene_3d"][s] for s in common],
                          [by_condition["text_only"][s] for s in common]),
        }
    return table


def summarize(results: Iterable[Dict]) -> Dict:
    """One call that produces the full result table for a run."""
    rs = list(results)
    return {
        "n_results": len(rs),
        "pipeline_health": pipeline_health(rs),
        "answer_accuracy": answer_accuracy(rs, group_by="condition"),
        "by_geover": answer_accuracy(rs, group_by="used_geover"),
        "condition_comparison": condition_comparison(rs),
        "verification_effect": verification_effect(rs),
        "relation_level": relation_level_accuracy(rs),
        "grounding_quality": grounding_quality(rs),
        "localization": localization_metrics(rs),
        "description": description_metrics(rs),
    }
