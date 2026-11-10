#!/usr/bin/env python3
"""PRISM-V command line entry point.

Four modes, mapping onto the framework's four claims:

    eval      the ablation matrix (task x input condition x verification on/off)
    analyze   tool-use strategy, reconstruction-quality and verification-mechanism sweeps
    verify    one question, printed with its full geometric audit trail
    smoke     the whole pipeline against a synthetic scene and a stub model, no GPU or API

``smoke`` exists so the wiring can be checked on a machine that has none of the perception
stack installed; it is the fastest way to confirm an edit did not break control flow.

Examples
--------
    # 1. reconstruct + cache geometry (inherited preprocessing, unchanged behaviour)
    python -m reconstruct_pipe --mode batch \\
        --jsonl_path datasets/MindCube/data/raw/MindCube_tinybench.jsonl \\
        --base_data_path datasets/MindCube/data --gpu_ids 0,1

    # 2. the headline ablation
    python run_prismv.py eval \\
        --jsonl_path datasets/MindCube/data/raw/MindCube_tinybench.jsonl \\
        --processed_dir output/preprocessed --tasks qa --max_entries 200 \\
        --output results/ablation.json

    # 3. three tasks at once
    python run_prismv.py eval --tasks qa localize describe ...

    # 4. diagnostics
    python run_prismv.py analyze --report results/ablation.json

    # 5. no dataset, no API key
    python run_prismv.py smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _build_mllm(args):
    """Pick the MLLM backend. Kept in one place so every mode agrees on what ran."""
    if args.stub:
        from prismv.pipeline import StubMLLM

        print("[PRISM-V] using the offline stub MLLM -- results are wiring checks, not "
              "measurements")
        return StubMLLM(bias=args.stub_answer)
    from prismv.pipeline import OpenAIMLLM

    return OpenAIMLLM(api_key=os.getenv("OPENAI_API_KEY"))


def _build_pipeline(args, mllm):
    from prismv.pipeline import PRISMVPipeline

    generator = None
    if not args.stub:
        from prismv.agent.code_query import generate_code_from_query
        from prismv.interface import Scene  # noqa: F401

        def generator(scene):
            return generate_code_from_query(scene, model=args.model)

    return PRISMVPipeline(
        mllm=mllm,
        code_generator=generator,
        processed_dir=args.processed_dir,
        verification_weight=args.verification_weight,
        min_support=args.min_support,
        conf_thres=args.conf_thres,
        exec_timeout=args.exec_timeout,
        fallback_clusters=args.fallback_clusters,
    )


def _load_records(args):
    from prismv.eval.datasets import load_mindcube, task_coverage

    records = load_mindcube(args.jsonl_path, base_data_path=args.base_data_path,
                            max_entries=args.max_entries, task=args.tasks[0])
    cov = task_coverage(records)
    print(f"[PRISM-V] loaded {len(records)} records; scoring coverage:")
    for task in ("qa", "localize", "describe"):
        c = cov.get(task, {})
        print(f"    {task:<10} n={c.get('n', 0)} options={c.get('with_options', 0)} "
              f"gt_answer={c.get('with_gt_answer', 0)} gt_entity={c.get('with_gt_entity', 0)} "
              f"gt_relations={c.get('with_gt_relations', 0)} "
              f"masks={c.get('with_entity_masks', 0)}")
    print("    (a task with no ground truth cannot be scored on it -- this is reported "
          "rather than papered over)")
    return records


# --------------------------------------------------------------------------- modes
def cmd_eval(args):
    from prismv.eval.runner import EvalRunner
    from prismv.pipeline import Condition

    mllm = _build_mllm(args)
    pipeline = _build_pipeline(args, mllm)
    records = _load_records(args)

    states = (False, True) if args.ablate_geover else (not args.no_geover,)
    conditions = Condition.ALL if args.all_conditions else (
        tuple(args.conditions) if args.conditions else (Condition.SCENE_3D,))
    runner = EvalRunner(pipeline, records=records, tasks=tuple(args.tasks),
                        conditions=conditions, verification_states=states,
                        progress_every=args.progress_every, limit=args.max_entries)
    runner.run(verbose=not args.quiet)
    report = runner.report(args.output, include_results=args.save_results)
    _print_report(report)
    return 0


def cmd_analyze(args):
    from prismv import analysis

    with open(args.report) as f:
        report = json.load(f)
    results = []
    for cell in report.get("cells", []):
        results.extend(cell.get("results", []))
    if not results:
        # An empty analysis is the most dangerous kind of output here: every downstream
        # table renders, every count is simply zero, and it reads as "the mechanism did
        # nothing" rather than "there was nothing to analyse".
        print("error: the report contains no per-question results, so there is nothing to "
              "analyse.\n"
              "      re-run the evaluation with --save_results, e.g.\n"
              f"      python run_prismv.py eval --save_results "
              f"--jsonl_path <bench.jsonl> --output {args.report}", file=sys.stderr)
        return 2
    out = {
        "source_report": args.report,
        "n_results": len(results),
        "predicate_vocabulary": analysis.predicate_vocabulary_coverage(results=results),
        "tool_strategy": analysis.tool_strategy_analysis(results),
        "verification_mechanism": analysis.verification_mechanism_analysis(results),
    }
    print(json.dumps(out, indent=2)[: args.max_print])
    if args.output:
        analysis.write_report(out, args.output)
        print(f"\n[PRISM-V] analysis -> {args.output}")
    return 0


def cmd_verify(args):
    """Verify one hand-written claim, printing the whole audit trail."""
    from prismv.interface import PRISM_V, Scene

    PRISM_V.PROCESSED_BASE_DIR = args.processed_dir
    scene = Scene(args.images or [], args.question, scene_id=args.scene_id)
    ents = PRISM_V.ground_entities(scene, conf_thres=args.conf_thres,
                                   fallback_clusters=args.fallback_clusters)
    print(f"grounded {len(ents)} entities via {ents.provenance.get('grounding_path')}")
    print(f"entity names: {ents.names}")
    if not len(ents):
        print("nothing grounded -- verification will abstain. Provenance: "
              f"{ents.provenance}")
        return 1
    if args.options:
        opts = {}
        for item in args.options:
            letter, _, text = item.partition("=")
            opts[letter.strip().upper()] = text.strip()
        res = PRISM_V.check_options(scene, args.question, opts, ref_view=args.ref_view,
                                    verification_weight=args.verification_weight,
                                    min_support=args.min_support)
        print(json.dumps(res, indent=2))
        print(f"\nanswer={res['answer']} applied={res['applied']} "
              f"ref_view={res['ref_view_inferred']} note={res['note']}")
    else:
        if not (args.relation and args.subject and args.object):
            print("--relation/--subject/--object are required without --options")
            return 2
        res = PRISM_V.verify(scene, args.relation, args.subject, args.object,
                             third=args.third, ref_view=args.ref_view)
        print(json.dumps(res, indent=2))
    return 0


def cmd_smoke(args):
    """Run the framework with no dataset, no weights and no API key."""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    rc = 0
    for script in ("tests/test_geover_synthetic.py", "tests/test_pipeline_offline.py"):
        print(f"\n=== {script} ===")
        p = subprocess.run([sys.executable, os.path.join(here, script)],
                           capture_output=True, text=True)
        tail = (p.stdout or "").strip().splitlines()[-3:]
        print("\n".join(tail))
        if p.returncode != 0:
            print(p.stdout[-2000:])
            print(p.stderr[-2000:])
            rc = 1
    print("\n[PRISM-V] smoke " + ("OK -- perceive/verify/reason wiring is sound"
                                  if rc == 0 else "FAILED"))
    return rc


# --------------------------------------------------------------------------- reporting
def _print_report(report):
    print("\n" + "=" * 74)
    print("PRISM-V evaluation")
    print("=" * 74)
    for cell in report.get("cells", []):
        a = cell["answer"]
        eff = cell["effect"]
        h = cell["health"]
        flag = "on " if cell["used_geover"] else "off"
        print(f"  {cell['task']:<9} {cell['condition']:<10} geover={flag} "
              f"n={a['n_evaluable']:>4}  acc={a['accuracy']:.4f} "
              f"[{a.get('lo', 0):.3f}, {a.get('hi', 0):.3f}]  "
              f"parse={h.get('parse_rate', 0):.2f} exec={h.get('execution_rate', 0):.2f} "
              f"fix={eff.get('fixes', 0)} dmg={eff.get('damages', 0)}")
    hl = report.get("headline", {})
    if hl:
        print("\n  Headline contrasts (paired bootstrap):")
        for k, v in hl.items():
            if "delta" in v:
                star = "*" if v["lo"] > 0 or v["hi"] < 0 else " "
                print(f"    {k:<34} delta={v['delta']:+.4f} "
                      f"CI[{v['lo']:+.4f}, {v['hi']:+.4f}] n={v.get('matched_n', v.get('n'))} "
                      f"{star}")
        print("    (* = interval excludes zero)")


# --------------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(
        description="PRISM-V: programmable 3D spatial reasoning with retrieval-based "
                    "geometric verification")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--jsonl_path", help="benchmark JSONL (MindCube by default)")
        sp.add_argument("--base_data_path", help="prefix for relative image paths")
        sp.add_argument("--processed_dir", default=None,
                        help="directory of cached reconstructions (reconstruct_pipe output)")
        sp.add_argument("--max_entries", type=int, default=None)
        sp.add_argument("--model", default="gpt-5")
        sp.add_argument("--stub", action="store_true",
                        help="use the offline stub MLLM (no API calls, not a measurement)")
        sp.add_argument("--stub_answer", default="A", help="option the stub always prefers")
        sp.add_argument("--conf_thres", type=float, default=0.0,
                        help="depth-confidence cut for mask lifting")
        sp.add_argument("--fallback_clusters", type=int, default=0,
                        help="cluster the point cloud when no masks exist (0 = abstain)")
        sp.add_argument("--exec_timeout", type=float, default=180.0)
        sp.add_argument("--verification_weight", type=float, default=1.0,
                        help="lambda on the verification term; 0 disables reranking")
        sp.add_argument("--min_support", type=int, default=1,
                        help="hard-supported constraints required before overriding the model")

    sp = sub.add_parser("eval", help="run the ablation matrix")
    common(sp)
    sp.add_argument("--tasks", nargs="+", default=["qa"],
                    choices=["qa", "localize", "describe"])
    sp.add_argument("--conditions", nargs="+", default=None,
                    choices=["text_only", "image_2d", "scene_3d"])
    sp.add_argument("--all_conditions", action="store_true",
                    help="run text_only / image_2d / scene_3d (the headline comparison)")
    sp.add_argument("--ablate_geover", action="store_true",
                    help="run both verification=off and verification=on")
    sp.add_argument("--no_geover", action="store_true", help="disable verification entirely")
    sp.add_argument("--output", default="results/prismv_eval.json")
    sp.add_argument("--save_results", action="store_true", help="embed per-question results")
    sp.add_argument("--progress_every", type=int, default=20)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("analyze", help="diagnostics over a finished eval report")
    sp.add_argument("--report", required=True)
    sp.add_argument("--output", default="results/prismv_analysis.json")
    sp.add_argument("--max_print", type=int, default=6000)
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("verify", help="verify one claim / one question, print the audit trail")
    sp.add_argument("--question", required=True)
    sp.add_argument("--images", nargs="*", default=None)
    sp.add_argument("--scene_id", default=None)
    sp.add_argument("--processed_dir", default=None)
    sp.add_argument("--relation", default=None)
    sp.add_argument("--subject", default=None)
    sp.add_argument("--object", dest="object", default=None)
    sp.add_argument("--third", default=None)
    sp.add_argument("--ref_view", type=int, default=None)
    sp.add_argument("--options", nargs="*", default=None,
                    help="e.g. A=cup B=plant C=sofa D=lamp")
    sp.add_argument("--conf_thres", type=float, default=0.0)
    sp.add_argument("--fallback_clusters", type=int, default=0)
    sp.add_argument("--verification_weight", type=float, default=1.0)
    sp.add_argument("--min_support", type=int, default=1)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("smoke", help="offline self-check, no GPU / no API key")
    sp.set_defaults(func=cmd_smoke)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
