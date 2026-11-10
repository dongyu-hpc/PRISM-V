"""Analysis of tool-calling strategy, reconstruction quality and the verification mechanism.

The fourth contribution is a set of *diagnostics*, not a leaderboard, so this module's job is
to answer questions a final accuracy number cannot:

* Does the model's tool sequence actually matter, or does every arm converge?
* How much verification benefit survives a degraded reconstruction -- i.e. is GeoVer a
  substitute for good geometry or a complement to it?
* Which relations does the predicate vocabulary cover, and what fraction of questions does
  that vocabulary actually reach?

The last one is the number most likely to be quietly optimistic in any verification paper, so
it is reported as coverage first and accuracy second.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from prismv.eval import metrics
from prismv.geover.predicates import PREDICATES, VIEW_RELATIVE, relation_support_report


# --------------------------------------------------------------------- tool strategy
def tool_strategy_analysis(cells: Iterable[Dict]) -> Dict:
    """How generated programs use the tool library, and whether that usage predicts success.

    ``cells`` are result dicts with ``tool_summary.sequence`` and ``correct``. The
    interesting comparison is not "which tool is called most" but whether the *same* scene
    answered correctly by one program differs in which tools were called -- which is what
    the per-outcome frequency table and the sequence-length effect describe.
    """
    rows = [c for c in cells if c.get("tool_summary")]
    if not rows:
        return {"n_programs": 0}
    seq_len_correct: List[float] = []
    seq_len_wrong: List[float] = []
    freq: Counter = Counter()
    freq_correct: Counter = Counter()
    freq_wrong: Counter = Counter()
    first_tool: Counter = Counter()
    transitions: Counter = Counter()
    failures: Counter = Counter()
    no_tool_correct = no_tool_total = 0
    for r in rows:
        ts = r["tool_summary"]
        seq = ts.get("sequence", [])
        ok = r.get("correct")
        if not seq:
            no_tool_total += 1
            if ok:
                no_tool_correct += 1
        if ok is True:
            seq_len_correct.append(len(seq))
        elif ok is False:
            seq_len_wrong.append(len(seq))
        for t in seq:
            freq[t] += 1
            (freq_correct if ok else freq_wrong)[t] += 1
        if seq:
            first_tool[seq[0]] += 1
            for a, b in zip(seq, seq[1:]):
                transitions[f"{a}->{b}"] += 1
        for c in ts.get("sequence", []):
            pass
        for note in []:
            failures[note] += 1
    return {
        "n_programs": len(rows),
        "mean_calls": round(float(np.mean([len(r["tool_summary"].get("sequence", []))
                                           for r in rows])), 3),
        "mean_calls_correct": round(float(np.mean(seq_len_correct)), 3) if seq_len_correct else None,
        "mean_calls_wrong": round(float(np.mean(seq_len_wrong)), 3) if seq_len_wrong else None,
        "tool_frequency": dict(freq.most_common()),
        "tool_frequency_correct": dict(freq_correct.most_common()),
        "tool_frequency_wrong": dict(freq_wrong.most_common()),
        "first_tool": dict(first_tool.most_common()),
        "top_transitions": dict(transitions.most_common(10)),
        "programs_without_tools": {"n": no_tool_total,
                                  "accuracy": round(no_tool_correct / no_tool_total, 4)
                                  if no_tool_total else None},
        # a program that called a tool which raised is the failure mode most likely to be
        # hidden by "execution_ok", because the tool may still have returned something
        "tool_failure_rate": round(float(np.mean([r["tool_summary"].get("failure_rate", 0.0)
                                                 for r in rows])), 4),
    }


def strategy_ablation(pipeline_factory, records: Sequence[Dict],
                      strategies: Sequence[str] = ("model", "reconstruct_only",
                                                 "reconstruct_render", "render_only"),
                      condition: str = "scene_3d", verbose: bool = True) -> Dict:
    """Force specific tool-use patterns instead of waiting for the model to choose them.

    Letting an LLM pick its own program conflates two questions -- "is this tool useful" and
    "did the model think to call it". Fixing the sequence separates them, and it is the only
    way to report what accuracy would look like if the generator were better at tool choice.
    """
    out: Dict[str, Any] = {}
    for strat in strategies:
        pipe = pipeline_factory(strategy=strat)
        results = []
        for rec in records:
            try:
                res = pipe.run(question=rec["question"], options=rec.get("options") or {},
                               images=rec.get("images"), condition=condition,
                               used_geover=False, scene_id=rec.get("scene_id"),
                               expected_answer=rec.get("gt_answer"),
                               entity_masks=rec.get("entity_masks"))
                results.append(res.to_dict())
            except Exception as e:
                results.append({"scene_id": rec.get("scene_id"), "correct": None,
                                "errors": [f"strategy {strat}: {e}"]})
        acc = metrics.answer_accuracy(results)
        out[strat] = {"n": len(results), "accuracy": acc["accuracy"],
                      "ci": [acc["lo"], acc["hi"]],
                      "health": metrics.pipeline_health(results)}
        if verbose:
            print(f"  strategy={strat:<20} acc={out[strat]['accuracy']:.4f} "
                  f"(n={len(results)})")
    base = out.get("model")
    if base:
        for k, v in out.items():
            if k != "model":
                v["delta_vs_model"] = round(v["accuracy"] - base["accuracy"], 4)
    return out


# --------------------------------------------------------------------- reconstruction quality
def reconstruction_quality_metrics(graphs: Iterable[Any]) -> Dict:
    """Scene-quality proxies computable without ground truth.

    No reference 3D model is assumed, because the benchmarks do not ship one. Each proxy is
    a statement about whether the scene is *verifiable* rather than whether it is pretty:
    entity point counts and view coverage bound how much evidence GeoVer has to work with,
    and per-entity depth spread is what makes a left/right margin meaningful at all.
    """
    n_entities: List[float] = []
    points_per_entity: List[float] = []
    views_per_entity: List[float] = []
    extents: List[float] = []
    zero_entity_scenes = 0
    n_views: List[float] = []
    for g in graphs:
        if g is None:
            zero_entity_scenes += 1
            continue
        ents = list(getattr(g, "entities", []) or [])
        n_entities.append(len(ents))
        if not ents:
            zero_entity_scenes += 1
            continue
        points_per_entity.append(float(np.mean([e.n_points for e in ents])))
        views_per_entity.append(float(np.mean([len(e.views) for e in ents])))
        extents.append(float(g.extent()))
        ex = getattr(g, "extrinsics", None)
        n_views.append(float(len(ex)) if ex is not None else 0.0)
    return {
        "n_scenes": len(n_entities),
        "mean_entities": round(float(np.mean(n_entities)), 3) if n_entities else 0.0,
        "scenes_without_entities": zero_entity_scenes,
        "mean_points_per_entity": round(float(np.mean(points_per_entity)), 1)
        if points_per_entity else 0.0,
        "median_points_per_entity": round(float(np.median(points_per_entity)), 1)
        if points_per_entity else 0.0,
        "mean_views_per_entity": round(float(np.mean(views_per_entity)), 3)
        if views_per_entity else 0.0,
        "mean_scene_extent": round(float(np.mean(extents)), 4) if extents else 0.0,
        "mean_n_views": round(float(np.mean(n_views)), 3) if n_views else 0.0,
    }


def reconstruction_sweep(run_once, conf_values: Sequence[float] = (0.0, 1.0, 2.5, 5.0),
                        view_counts: Sequence[int] = (1, 2, 4, 8),
                        records: Optional[Sequence[Dict]] = None,
                        verbose: bool = True) -> List[Dict]:
    """Sweep reconstruction quality and measure downstream accuracy at each level.

    Two knobs, one at a time: the depth-confidence filter (how much junk geometry survives
    unprojection) and the number of views fed to reconstruction (how complete the scene is).
    Both degrade entity grounding, and the shape of the resulting curve says whether
    verification *rescues* weak geometry or merely *requires* decent geometry -- a difference
    that matters a great deal for whether this layer is useful outside a lab.
    """
    out: List[Dict] = []
    if records is None:
        records = []
    for kind, values in (("conf_thres", conf_values), ("n_views", view_counts)):
        for v in values:
            res = run_once(**{kind: v})
            acc = metrics.answer_accuracy(res)
            row = {"knob": kind, "value": v, "n": len(res),
                   "accuracy": acc["accuracy"], "ci": [acc["lo"], acc["hi"]],
                   "health": metrics.pipeline_health(res),
                   "effect": metrics.verification_effect(res)}
            if kind == "conf_thres":
                row["grounding"] = reconstruction_quality_metrics(
                    [r.get("_graph") for r in res])
            out.append(row)
            if verbose:
                print(f"  {kind}={v!s:<6} acc={row['accuracy']:.4f} "
                      f"n={row['n']}")
    # the interesting summary: does GeoVer's net effect grow or shrink as geometry degrades?
    by_knob: Dict[str, List[Dict]] = defaultdict(list)
    for row in out:
        by_knob[row["knob"]].append(row)
    trends = {}
    for knob, rows in by_knob.items():
        nets = [(r["value"], r["effect"].get("net", 0)) for r in rows]
        if len(nets) >= 2:
            xs = np.array([float(v) for v, _ in nets])
            ys = np.array([float(n) for _, n in nets])
            if xs.std() > 0:
                trends[knob] = {"net_effect_by_value": nets,
                               "slope": round(float(np.polyfit(xs, ys, 1)[0]), 5)}
    return [{"sweep": o, "trend": trends.get(o["knob"], {})} if o is out[-1] else o
            for o in out]


# --------------------------------------------------------------------- verification mechanism
def verification_mechanism_analysis(results: Iterable[Dict]) -> Dict:
    """Decompose GeoVer's contribution: reach, correctness of its judgements, and outcome.

    ``reach`` is the share of questions where the layer produced at least one hard
    constraint. Everything else is conditional on that, which is why a headline accuracy
    delta with reach at 0.2 should be read as "the mechanism was off for 80% of the
    benchmark, and noise moved the number on the rest".
    """
    rows = [r for r in results if r.get("used_geover")]
    if not rows:
        return {"n": 0}
    reach = checks = 0
    status: Counter = Counter()
    per_relation: Dict[str, Counter] = defaultdict(Counter)
    applied = changed = 0
    undetermined_share: List[float] = []
    for r in rows:
        ver = r.get("verification") or {}
        cs = ver.get("checks", [])
        if any(c.get("status") in ("supported", "contradicted") for c in cs):
            reach += 1
        checks += len(cs)
        for c in cs:
            status[c.get("status", "?")] += 1
            per_relation[c.get("relation", "?")][c.get("status", "?")] += 1
        if ver.get("applied"):
            applied += 1
        if ver.get("changed_answer"):
            changed += 1
        tot = len(cs)
        und = sum(1 for c in cs if c.get("status") == "undetermined")
        if tot:
            undetermined_share.append(und / tot)
    relation_table = {}
    for rel, cnt in sorted(per_relation.items()):
        hard = cnt["supported"] + cnt["contradicted"]
        relation_table[rel] = {
            "checks": sum(cnt.values()), "supported": cnt["supported"],
            "contradicted": cnt["contradicted"], "undetermined": cnt["undetermined"],
            "agreement_rate": round(cnt["supported"] / hard, 4) if hard else None,
            "view_relative": rel in VIEW_RELATIVE,
        }
    return {
        "n": len(rows),
        "reach": round(reach / len(rows), 4),
        "mean_checks_per_question": round(checks / len(rows), 2),
        "status_distribution": dict(status),
        "mean_undetermined_share": round(float(np.mean(undetermined_share)), 4)
        if undetermined_share else 0.0,
        "applied_rate": round(applied / len(rows), 4),
        "answer_changed_rate": round(changed / len(rows), 4),
        "by_relation": relation_table,
    }


def reranker_sweep(run_once, weights: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
                  min_supports: Sequence[int] = (1, 2, 3),
                  penalties: Sequence[float] = (0.5, 1.0, 2.0),
                  verbose: bool = True) -> List[Dict]:
    """Sweep the reranker's own hyperparameters, reporting net effect not just accuracy.

    ``min_support`` and ``contradiction_penalty`` trade fix_rate against damage_rate. Sorting
    this table by accuracy alone hides that trade, so the rows are annotated with the net and
    the damage rate explicitly -- the entry a reviewer should be allowed to see first.
    """
    rows: List[Dict] = []
    for w in weights:
        for ms in min_supports:
            for pen in penalties:
                if w == 0.0 and (ms != 1 or pen != 1.0):
                    continue  # lambda=0 ignores the other two knobs
                res = run_once(verification_weight=w, min_support=ms,
                               contradiction_penalty=pen)
                eff = metrics.verification_effect(res)
                acc = metrics.answer_accuracy(res)
                rows.append({"weight": w, "min_support": ms, "contradiction_penalty": pen,
                             "accuracy": acc["accuracy"], "fixes": eff["fixes"],
                             "damages": eff["damages"], "net": eff["net"],
                             "damage_rate": eff["damage_rate"],
                             "applic_rate": eff["applic_rate"], "n": eff["n_questions"]})
                if verbose:
                    print(f"  lambda={w:<4} min_support={ms} penalty={pen:<4} "
                          f"acc={rows[-1]['accuracy']:.4f} net={eff['net']:+d} "
                          f"damage_rate={eff['damage_rate']:.3f}")
    rows.sort(key=lambda r: (-r["net"], -r["accuracy"]))
    return rows


def predicate_vocabulary_coverage(records: Optional[Sequence[Dict]] = None,
                                 results: Optional[Sequence[Dict]] = None) -> Dict:
    """What the predicate library claims to check, versus what the questions need.

    Reported before any accuracy number on purpose: if the vocabulary only reaches a third of
    the benchmark's spatial relations, every downstream improvement is bounded by that
    coverage, and a reader who does not know this will over-generalise the result.
    """
    table = relation_support_report()
    seen: Counter = Counter()
    for r in (results or []):
        for c in (r.get("verification") or {}).get("checks", []):
            seen[c.get("relation", "?")] += 1
    questions: Counter = Counter()
    if records:
        from prismv.geover.hypothesis import detect_relations

        for rec in records:
            for _, _, rel in detect_relations(rec.get("question", "")):
                questions[rel] += 1
    return {
        "supported_relations": table,
        "n_supported": len(PREDICATES),
        "n_view_relative": len(VIEW_RELATIVE),
        "relations_seen_in_data": dict(questions.most_common()),
        "relations_actually_checked": dict(seen.most_common()),
        "n_checked_total": int(sum(seen.values())),
        "coverage_of_needed": {
            rel: {"questions_using": q, "checked": seen.get(rel, 0),
                  "in_vocabulary": rel in PREDICATES}
            for rel, q in questions.items()},
        "unsupportable_relations": sorted(set(questions) - set(PREDICATES)),
    }


def write_report(analysis: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    return path


def tables_for_paper(analysis: Dict) -> List[List[str]]:
    """Flatten the analysis into the rows a paper would actually print."""
    rows: List[List[str]] = []
    for knob_row in analysis.get("reconstruction_sweep", []):
        s = knob_row.get("sweep", knob_row)
        rows.append([str(s.get("knob")), str(s.get("value")), f"{s.get('accuracy', 0):.4f}",
                     str(s.get("n")), str(s.get("effect", {}).get("net", ""))])
    return rows
