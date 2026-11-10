"""Offline end-to-end test for the PRISM-V pipeline, evaluation runner and analysis module.

Nothing here needs a GPU, model weights, trimesh/open3d, or an API key. The scene is written
to disk as entity point sets plus camera matrices -- exactly the artefacts the reconstruction
stage produces -- and the MLLM is the deterministic stub. What is under test is PRISM-V's
*control flow*, which is the part that breaks silently:

  condition routing (text_only / image_2d / scene_3d)
  program execution and tool-call recording
  grounding precedence (cached geometry used, no redundant reconstruction)
  verification reaching the final answer, and a measurable net effect on accuracy
  the three evaluation tasks and the ablation matrix plumbing
  metrics and analysis computed over real result dicts

If this passes, a live run differs only in the backend, not in the wiring.

Run:  python tests/test_pipeline_offline.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prismv import analysis  # noqa: E402
from prismv.eval import metrics  # noqa: E402
from prismv.eval.datasets import parse_options, to_task_record  # noqa: E402
from prismv.eval.runner import EvalRunner  # noqa: E402
from prismv.interface import PRISM_V, Scene  # noqa: E402
from prismv.pipeline import Condition, PRISMVPipeline, StubMLLM, parse_code  # noqa: E402

RNG = np.random.RandomState(11)
TMP = os.path.join(tempfile.gettempdir(), "prismv_offline_scene")

# Same layout as tests/test_geover_synthetic.py: camera 0 at the origin looking down +Z,
# +X to the right, +Y downward. Camera 1 is turned around, which flips left/right.
LAYOUT = {
    "book": (-0.4, 0.1, 2.0),
    "cup": (0.3, 0.1, 2.0),
    "plant": (-0.6, 0.1, 2.2),
    "lamp": (0.0, -0.5, 3.0),
    "sofa": (0.0, 0.2, 1.2),
}

# "left of the book" has exactly one geometrically endorsed option here: plant (x=-0.6) is
# left of book (x=-0.4), while cup, sofa and lamp all sit at or right of it and are therefore
# contradicted. The "right of" variant endorses cup, sofa AND lamp simultaneously -- a
# genuinely ambiguous item, which makes it the wrong fixture for asserting a forced flip. It
# gets its own test below instead, because that ambiguity is a real property of spatial
# benchmarks and bounds what reranking can possibly achieve.
QUESTION = "From the viewpoint presented in image 1, what is to the left of the book?"
AMBIGUOUS_QUESTION = "From the viewpoint presented in image 1, what is to the right of the book?"
OPTIONS = {"A": "cup", "B": "plant", "C": "sofa", "D": "lamp"}
GT = "B"

# What a real code generator would emit for this question: build the scene and enumerate what
# could be grounded. Deliberately avoids reconstruct()/rendering so the test stays CPU-free.
PROGRAM = (
    "def program(input_scene):\n"
    "    entities = PRISM_V.ground_entities(input_scene)\n"
    "    return 'grounded: ' + ', '.join(entities.names)\n"
)


def _cameras():
    identity = np.eye(4)
    R = np.diag([-1.0, 1.0, -1.0])
    origin = np.array([0.0, 0.0, 4.0])
    turned = np.eye(4)
    turned[:3, :3] = R
    turned[:3, 3] = -(R @ origin)
    return np.stack([identity, turned])


def write_scene(n_points: int = 300) -> str:
    """Materialise the processed-scene artefacts GeoVer reads from disk."""
    global RNG
    RNG = np.random.RandomState(11)
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    ents = []
    for name, center in LAYOUT.items():
        pts = np.asarray(center) + RNG.randn(n_points, 3) * 0.05
        ents.append({"name": name, "points": pts.tolist(), "views": [0, 1],
                     "pixel_counts": [n_points, n_points], "confidence": 1.0})
    with open(os.path.join(TMP, "entities.json"), "w") as f:
        json.dump({"extrinsics": _cameras().tolist(), "entities": ents}, f)
    return TMP


def make_pipeline(mllm, processed_dir, **kw):
    return PRISMVPipeline(mllm=mllm, code_generator=lambda s: PROGRAM,
                          processed_dir=processed_dir, **kw)


def _qa_result(pipeline, condition, used_geover, mllm_answer="A", gt=None):
    """Run one QA item; ``mllm_answer`` is forced so each arm states the model's belief."""
    pipeline.mllm.bias = mllm_answer
    return pipeline.run(question=QUESTION, options=OPTIONS, images=[], condition=condition,
                        used_geover=used_geover, scene_id="s1",
                        expected_answer=GT if gt is None else gt).to_dict()


def _records():
    return [to_task_record({
        "id": "s1",
        "question": QUESTION + "\nA. cup\nB. plant\nC. sofa\nD. lamp",
        "images": [], "gt_answer": GT,
    }, task="qa")]


# ----------------------------------------------------------------- perception / grounding
def test_scene_artefacts_load_without_perception_stack():
    """Grounding must work from cached geometry alone -- no torch, no trimesh, no open3d."""
    d = write_scene()
    PRISM_V.PROCESSED_BASE_DIR = d
    scene = Scene([], QUESTION, scene_id=os.path.basename(d))
    ents = PRISM_V.ground_entities(scene)
    assert len(ents) == len(LAYOUT), ents
    assert scene.reconstruction is None, "must not reconstruct when geometry is already cached"
    assert ents.provenance["grounding_path"] == "entity_points", ents.provenance
    assert ents.provenance.get("gravity_estimated") is True
    print(f"  grounded {len(ents)} entities from disk via "
          f"{ents.provenance['grounding_path']}, reconstruction never invoked")


def test_both_cache_layouts_resolve():
    """``BASE/<scene_id>/`` (batch) and ``BASE/`` itself (single-scene) both work.

    The fallback exists so a machine without the perception stack can still verify against
    cached geometry; covering only one shape would leave half of that untested, and the
    untested half is the shape the batch preprocessing actually emits.
    """
    d = write_scene()
    PRISM_V.PROCESSED_BASE_DIR = d
    flat = Scene([], QUESTION, scene_id="not_a_real_subdirectory")
    assert len(PRISM_V.ground_entities(flat)) == len(LAYOUT), "BASE/ fallback failed"

    batch_root = tempfile.mkdtemp(prefix="prismv_batch_")
    sid = "scene_42"
    os.makedirs(os.path.join(batch_root, sid))
    for fn in os.listdir(d):
        shutil.copy(os.path.join(d, fn), os.path.join(batch_root, sid, fn))
    PRISM_V.PROCESSED_BASE_DIR = batch_root
    nested = Scene([], QUESTION, scene_id=sid)
    ents = PRISM_V.ground_entities(nested)
    assert len(ents) == len(LAYOUT), "BASE/<scene_id>/ resolution failed"
    assert ents.provenance["grounding_path"] == "entity_points"
    shutil.rmtree(batch_root, ignore_errors=True)
    print("  both cache layouts resolve: BASE/ and BASE/<scene_id>/")


def test_grounding_degrades_without_perception_stack():
    """No cached geometry and no perception stack must yield an empty set plus a reason.

    This is the difference between "GeoVer abstained", which the metrics can count, and "the
    run crashed", which silently shrinks the benchmark.
    """
    empty_dir = tempfile.mkdtemp(prefix="prismv_nogeo_")
    PRISM_V.PROCESSED_BASE_DIR = empty_dir
    scene = Scene([], QUESTION, scene_id="nothing_cached_here")
    ents = PRISM_V.ground_entities(scene)
    assert len(ents) == 0, ents
    assert bool(ents.provenance.get("perception_error")), ents.provenance
    p = make_pipeline(StubMLLM(bias="A"), empty_dir)
    res = p.run(question=QUESTION, options=OPTIONS, images=[], condition=Condition.SCENE_3D,
                used_geover=True, scene_id="nothing_cached_here",
                expected_answer=GT).to_dict()
    assert res["final_answer"] == "A", "no geometry must not move the answer"
    assert res["verification"].get("note") == "no grounded entities", res["verification"]
    assert res["errors"] == [], res["errors"]
    shutil.rmtree(empty_dir, ignore_errors=True)
    print(f"  degraded grounding: 0 entities, reason recorded, "
          f"pipeline still answered ({res['verification']['note']})")


# ----------------------------------------------------------------- program execution
def test_program_execution_and_tool_recording():
    from prismv.pipeline import ToolRecorder, execute_program

    d = write_scene()
    PRISM_V.PROCESSED_BASE_DIR = d
    scene = Scene([], QUESTION, scene_id=os.path.basename(d))
    rec = ToolRecorder()
    fenced = parse_code("prose\n```python\n" + PROGRAM + "\n```\nmore prose")
    bare = parse_code(PROGRAM)
    assert fenced.startswith("def program"), fenced[:40]
    assert bare.startswith("def program"), bare[:40]
    assert parse_code("I cannot solve this.") == ""
    clue = execute_program(PROGRAM, scene, rec, timeout_s=10)
    assert isinstance(clue, str) and "cup" in clue, clue
    summary = rec.summary()
    assert summary["n_calls"] >= 1 and "ground_entities" in summary["unique_tools"], summary
    assert summary["failure_rate"] == 0.0, summary
    print(f"  program executed via fenced/bare forms, tools={summary['sequence']}")


def test_program_timeout_is_bounded():
    """Generated code is arbitrary; one runaway program must not stall a sweep."""
    d = write_scene()
    scene = Scene([], QUESTION, scene_id=os.path.basename(d))
    loop = "def program(input_scene):\n    while True:\n        pass\n"
    try:
        from prismv.pipeline import execute_program

        execute_program(loop, scene, timeout_s=1)
    except TimeoutError:
        print("  runaway program raised TimeoutError after 1s")
        return
    except Exception as e:  # signal-based abort may surface differently per platform
        print(f"  runaway program aborted with {type(e).__name__}")
        return
    raise AssertionError("infinite program was not interrupted")


# ----------------------------------------------------------------- condition routing
def test_text_only_and_image_2d_conditions_skip_verification():
    """The two baseline conditions must genuinely see no 3D and receive no rerank."""
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="A"), d)
    for cond in (Condition.TEXT_ONLY, Condition.IMAGE_2D):
        r = _qa_result(p, cond, used_geover=True)
        assert r["condition"] == cond
        assert r["verification"] == {}, f"{cond} must not run GeoVer: {r['verification']}"
        assert r["parse_ok"] is False, f"{cond} should not generate a program"
        assert r["final_answer"] == "A" and r["correct"] is False, r
        assert r["tool_summary"] == {}, r["tool_summary"]
    print("  text_only / image_2d routes: no program, no verification, answer untouched")


# ----------------------------------------------------------------- the headline behaviour
def test_verification_flips_a_wrong_answer_end_to_end():
    """The claim PRISM-V makes, measured on the whole pipeline, not on GeoVer in isolation."""
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="A"), d)
    off = _qa_result(p, Condition.SCENE_3D, used_geover=False)
    on = _qa_result(p, Condition.SCENE_3D, used_geover=True)

    assert off["final_answer"] == "A" and on["final_answer"] == GT, (
        off["final_answer"], on["final_answer"])
    assert off["correct"] is False and on["correct"] is True
    ver = on["verification"]
    assert ver["applied"] is True, ver["note"]
    assert ver["changed_answer"] is True
    assert ver["counts"]["supported"] >= 1 and ver["counts"]["contradicted"] >= 1, ver["counts"]
    # the audit trail has to show *why*: plant endorsed, cup contradicted, by named evidence
    cov = ver["option_coverage"]
    assert cov[GT]["supported"] >= 1 and cov["A"]["contradicted"] >= 1, cov
    for chk in ver["checks"]:
        assert chk["entities_used"] and chk["detail"], chk
    assert on["reasoning"].startswith("[stub]"), "first-pass reasoning must survive"
    # and the program really executed this time, with its tool call on the record
    assert on["parse_ok"] and on["execution_ok"], (on["errors"], on["parse_ok"])
    assert "ground_entities" in on["tool_summary"]["sequence"], on["tool_summary"]
    print(f"  A(wrong) -> {GT}(right): supported={ver['counts']['supported']} "
          f"contradicted={ver['counts']['contradicted']} "
          f"tools={on['tool_summary']['sequence']}")


def test_ambiguous_question_makes_the_layer_defer_not_guess():
    """Several options endorsed at once is a real property of spatial benchmarks.

    "Right of the book" admits cup, sofa and lamp here. With the model's belief on a
    contradicted option and the three endorsers equally weighted by the prior, the layer has
    no basis to choose between them and must decline to rerank rather than manufacture a
    winner. The metrics have to report how often that happens -- otherwise a gain measured on
    the decisive subset reads as a gain over the whole benchmark.
    """
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="B"), d)
    r = p.run(question=AMBIGUOUS_QUESTION, options=OPTIONS, images=[],
              condition=Condition.SCENE_3D, used_geover=True, scene_id="s_amb",
              expected_answer="A").to_dict()
    cov = r["verification"]["option_coverage"]
    endorsed = [l for l, c in cov.items() if c["supported"] > 0 and c["contradicted"] == 0]
    assert len(endorsed) >= 2, cov
    assert "B" not in endorsed, cov
    assert r["verification"]["applied"] is False, r["verification"]
    assert r["final_answer"] == "B", "the prior must stand when geometry cannot decide"
    eff = metrics.verification_effect([r])
    assert eff["n_ambiguous"] >= 1, eff
    assert eff["constraint_set_decisive_rate"] < 1.0, eff
    print(f"  {endorsed} all endorsed -> deferred; "
          f"decisive_rate={eff['constraint_set_decisive_rate']} "
          f"n_ambiguous={eff['n_ambiguous']}")


def test_partially_decisive_geometry_still_demotes_the_impossible():
    """Ambiguity among several options does not make the contradicted one safe.

    When the model's own pick is the geometrically impossible one, the layer can still act:
    not by choosing a winner, but by removing the answer the reconstruction rules out. This
    is the common real case, and it is why the reranker separates "cannot rank" from "cannot
    say anything".
    """
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="B"), d)
    # bias B = plant, which is left of the book; the question asks for right of the book
    r = p.run(question=AMBIGUOUS_QUESTION, options={"A": "cup", "B": "plant"}, images=[],
              condition=Condition.SCENE_3D, used_geover=True, scene_id="s_demote",
              expected_answer="A").to_dict()
    assert r["verification"]["option_coverage"]["B"]["contradicted"] >= 1, r["verification"]
    assert r["verification"]["applied"] is True, r["verification"]
    assert r["final_answer"] == "A" and r["correct"] is True, r["final_answer"]
    print("  contradicted option removed even though only two candidates remained")


def test_verification_can_also_be_harmless():
    """When the model was already right, the layer must not break it."""
    d = write_scene()
    p = make_pipeline(StubMLLM(bias=GT), d)
    r = _qa_result(p, Condition.SCENE_3D, used_geover=True, mllm_answer=GT)
    assert r["final_answer"] == GT and r["correct"] is True, r
    assert r["verification"]["changed_answer"] is False
    print("  already-correct answer preserved")


def test_verification_records_provenance_and_timings():
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="A"), d)
    r = _qa_result(p, Condition.SCENE_3D, True)
    assert r["provenance"]["grounding_path"] == "entity_points", r["provenance"]
    assert r["provenance"]["n_entities_grounded"] == len(LAYOUT)
    for stage in ("perceive", "reason", "verify"):
        assert stage in r["timings"] and r["timings"][stage] >= 0.0, r["timings"]
    assert r["clue_type"] == "str", r["clue_type"]
    print(f"  provenance + per-stage timings OK ({sorted(r['timings'])})")


def test_reference_frame_change_flips_the_verdict_end_to_end():
    """Asking about image 2 instead of image 1 must reverse which option geometry endorses."""
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="A"), d)
    q1 = "From the viewpoint presented in image 1, what is to the left of the book?"
    q2 = "From the viewpoint presented in image 2, what is to the left of the book?"
    for q, expected_view in ((q1, 0), (q2, 1)):
        r = p.run(question=q, options=OPTIONS, images=[], condition=Condition.SCENE_3D,
                  used_geover=True, scene_id="s_view", expected_answer=GT).to_dict()
        ver = r["verification"]
        assert ver["ref_view_inferred"] == expected_view, ver
        cov = ver["option_coverage"]
        endorsed = {l for l, c in cov.items() if c["supported"] > 0 and c["contradicted"] == 0}
        contradicted = {l for l, c in cov.items() if c["contradicted"] > 0}
        if expected_view == 0:
            # plant (x=-0.6) is the only thing left of the book (x=-0.4)
            assert endorsed == {"B"}, cov
            assert {"A", "C", "D"} <= contradicted, cov
        else:
            # turned around, +X is to the viewer's left, so the endorsement set inverts:
            # plant becomes contradicted and cup/sofa/lamp become the things "to the left"
            assert "B" in contradicted, cov
            assert {"A", "C", "D"} <= endorsed, cov
    print("  same scene, opposite viewer: endorsed set went {B} -> {A,C,D}, "
          "i.e. the verdict actually tracks the reference frame")


# ----------------------------------------------------------------- evaluation matrix
def test_eval_runner_produces_the_full_matrix():
    d = write_scene()
    runner = EvalRunner(make_pipeline(StubMLLM(bias="A"), d), records=_records(),
                        conditions=Condition.ALL, verification_states=(False, True),
                        tasks=("qa",), progress_every=0)
    cells = runner.run(verbose=False)
    assert len(cells) == 6, [c.key for c in cells]
    for c in cells:
        assert len(c.results) == 1, f"{c.key} silently produced {len(c.results)} results"
    keys = {c.key for c in cells}
    for cond in Condition.ALL:
        assert f"qa/{cond}/geover=off" in keys and f"qa/{cond}/geover=on" in keys, keys
    report = runner.report()
    on = {c["condition"]: c for c in report["cells"] if c["used_geover"]}
    off = {c["condition"]: c for c in report["cells"] if not c["used_geover"]}
    assert on["scene_3d"]["answer"]["accuracy"] == 1.0, on["scene_3d"]["answer"]
    assert off["scene_3d"]["answer"]["accuracy"] == 0.0, off["scene_3d"]["answer"]
    # the baselines have no 3D to verify against, so turning GeoVer on cannot move them
    for cond in (Condition.TEXT_ONLY, Condition.IMAGE_2D):
        assert on[cond]["answer"]["accuracy"] == off[cond]["answer"]["accuracy"], cond
    eff = on["scene_3d"]["effect"]
    assert eff["fixes"] == 1 and eff["damages"] == 0, eff
    assert eff["n_uniquely_endorsed"] == 1 and eff["n_ambiguous"] == 0, eff
    assert "geover_effect_on_scene_3d" in report["headline"], sorted(report["headline"])
    assert "scene_3d_vs_image_2d" in report["headline"]
    health = on["scene_3d"]["health"]
    assert health["parse_rate"] == 1.0 and health["execution_rate"] == 1.0, health
    assert health["grounded_rate"] == 1.0, health
    print(f"  6 cells, 1 result each; scene_3d acc off=0.0 on=1.0; "
          f"headline={sorted(report['headline'])}")


def test_condition_comparison_is_paired_and_flags_matching():
    """A comparison over different question subsets must be visibly flagged, not silent."""
    d = write_scene()
    runner = EvalRunner(make_pipeline(StubMLLM(bias="A"), d), records=_records(),
                        conditions=(Condition.SCENE_3D, Condition.IMAGE_2D),
                        verification_states=(True,), tasks=("qa",), progress_every=0)
    runner.run(verbose=False)
    # cross-condition contrasts live in the headline block; a single cell holds one condition
    pair = runner.report()["headline"]["scene_3d_vs_image_2d"]
    assert pair["matched_n"] == 1 and pair["matched"] is True, pair
    assert {"delta", "lo", "hi", "p_gt_zero"} <= set(pair), pair
    assert pair["delta"] == 1.0, pair
    print(f"  paired scene_3d - image_2d = {pair['delta']:+.2f} "
          f"CI[{pair['lo']}, {pair['hi']}] over n={pair['matched_n']}")


# ----------------------------------------------------------------- the other two tasks
def test_localization_task_scores_end_to_end():
    d = write_scene()
    rec = to_task_record({"id": "s1",
                          "question": "From the viewpoint presented in image 1, which object "
                                      "is to the right of the book?",
                          "images": [], "gt_answer": "A", "gt_entity": "cup",
                          "options": {"A": "cup", "B": "plant"}}, task="localize")
    assert rec["gt_entity"] == "cup", rec
    runner = EvalRunner(make_pipeline(StubMLLM(bias="A"), d), records=[rec],
                        conditions=(Condition.SCENE_3D,), verification_states=(True,),
                        tasks=("localize",), progress_every=0)
    cells = runner.run(verbose=False)
    assert len(cells[0].results) == 1
    res = cells[0].results[0]
    assert res["relation"] == "right_of" and res["anchor"] == "book", res
    assert res["localized"], "expected a ranked candidate list"
    # cup is furthest right of the book, so it must outrank plant (which is left of it)
    top = res["localized"][0]
    assert top["candidate"] == "cup", res["localized"][:2]
    assert top["margin"] > 0 and res["localized"][-1]["candidate"] != "cup"
    assert res["correct"] is True
    m = metrics.localization_metrics(cells[0].results)
    assert m["localization_accuracy_at_1"] == 1.0, m
    print(f"  localization: top-1={top['candidate']} margin={top['margin']:+.3f} "
          f"acc@1={m['localization_accuracy_at_1']}")


def test_localization_target_left_unset_when_unmatchable():
    """An option naming no reconstructed entity must not be coerced into a target."""
    rec = to_task_record({"id": "s9", "question": "which object is to the right of the book?",
                          "images": [], "gt_answer": "A",
                          "options": {"A": "a giraffe", "B": "a comet"}}, task="localize")
    assert rec["gt_entity"] is None, rec
    print("  unmatchable localisation target stays unresolved rather than guessed")


def test_description_task_scores_geometric_precision():
    d = write_scene()
    mllm = StubMLLM(scripted_text="The cup is to the right of the book. "
                                  "The lamp is above the sofa. "
                                  "The plant is to the right of the book.")
    runner = EvalRunner(make_pipeline(mllm, d),
                        records=[to_task_record({"id": "s1", "question": QUESTION,
                                                 "images": []}, task="describe")],
                        conditions=(Condition.SCENE_3D,), verification_states=(True,),
                        tasks=("describe",), progress_every=0)
    cells = runner.run(verbose=False)
    res = cells[0].results[0]
    checks = res["verification"]["checks"]
    assert len(checks) >= 3, checks
    rels = {(c["relation"], c["subject"], c["object"]) for c in checks}
    assert ("right_of", "cup", "book") in rels, rels
    assert ("above", "lamp", "sofa") in rels, rels
    # two claims are geometrically supported and one is false: that is the whole point of the
    # task, since string-overlap scoring cannot see the false one at all
    false = [c for c in checks if c["subject"] == "plant"]
    assert false and false[0]["status"] == "contradicted", false
    m = metrics.description_metrics(cells[0].results)
    assert m["n_scored"] == 1, m
    assert abs(m["geometric_precision"] - 2 / 3) < 0.02, m
    assert abs(m["contradiction_rate"] - 1 / 3) < 0.02, m
    print(f"  description: {len(checks)} claims checked, precision={m['geometric_precision']}, "
          f"contradiction_rate={m['contradiction_rate']}")


# ----------------------------------------------------------------- analysis module
def test_analysis_reports_reach_before_accuracy():
    """Vocabulary coverage bounds everything else, so it has to be stated up front."""
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="A"), d)
    results = [_qa_result(p, Condition.SCENE_3D, True)]
    mech = analysis.verification_mechanism_analysis(results)
    assert mech["n"] == 1 and mech["reach"] == 1.0, mech
    assert mech["applied_rate"] == 1.0 and mech["answer_changed_rate"] == 1.0, mech
    assert "left_of" in mech["by_relation"], mech["by_relation"]
    assert mech["by_relation"]["left_of"]["view_relative"] is True
    cov = analysis.predicate_vocabulary_coverage(
        records=[{"question": "what is to the left of the cup, and is the lamp above it?"}],
        results=results)
    assert cov["n_supported"] == len(cov["supported_relations"]) >= 11, cov
    assert "left_of" in cov["relations_seen_in_data"], cov["relations_seen_in_data"]
    assert cov["relations_seen_in_data"]["above"] == 1, cov
    assert cov["unsupportable_relations"] == [], cov
    print(f"  reach={mech['reach']}, vocabulary={cov['n_supported']} relations, "
          f"unsupportable={cov['unsupportable_relations']}")


def test_tool_strategy_analysis():
    d = write_scene()
    p = make_pipeline(StubMLLM(bias="A"), d)
    results = [_qa_result(p, Condition.SCENE_3D, True),
               _qa_result(p, Condition.SCENE_3D, False)]
    ts = analysis.tool_strategy_analysis(results)
    assert ts["n_programs"] == 2, ts
    assert "ground_entities" in ts["tool_frequency"], ts
    assert ts["first_tool"] == {"ground_entities": 2}, ts
    assert ts["mean_calls"] == 1.0, ts
    assert ts["programs_without_tools"]["n"] == 0, ts
    print(f"  tool strategy: freq={ts['tool_frequency']}, first={ts['first_tool']}, "
          f"mean_calls={ts['mean_calls']}")


def test_reconstruction_quality_metrics_respond_to_degradation():
    from prismv.geover.grounding import build_scene_graph

    d = write_scene()
    graph, _ = build_scene_graph(processed_dir=d)
    q = analysis.reconstruction_quality_metrics([graph])
    assert q["n_scenes"] == 1 and q["mean_entities"] == float(len(LAYOUT)), q
    assert q["mean_points_per_entity"] == 300.0, q
    assert q["scenes_without_entities"] == 0
    d2 = write_scene(n_points=40)
    g2, _ = build_scene_graph(processed_dir=d2)
    q2 = analysis.reconstruction_quality_metrics([g2])
    assert q2["mean_points_per_entity"] == 40.0 < q["mean_points_per_entity"], q2
    print(f"  reconstruction quality tracks degradation: points/entity "
          f"{q['mean_points_per_entity']} -> {q2['mean_points_per_entity']}")


# ----------------------------------------------------------------- metric hygiene
def test_metrics_are_honest_about_the_no_ground_truth_case():
    """Questions with no label must be counted as unevaluable, never scored wrong."""
    rows = [{"scene_id": "a", "condition": "scene_3d", "used_geover": True, "correct": True},
            {"scene_id": "b", "condition": "scene_3d", "used_geover": True, "correct": False},
            {"scene_id": "c", "condition": "scene_3d", "used_geover": True, "correct": None}]
    acc = metrics.answer_accuracy(rows)
    assert acc["n_evaluable"] == 2 and acc["n_skipped_no_gt"] == 1, acc
    assert acc["accuracy"] == 0.5 and acc["n"] == 2, acc
    assert acc["lo"] <= 0.5 <= acc["hi"], acc
    h = metrics.pipeline_health(rows)
    assert h["n"] == 3 and h["evaluable_rate"] == round(2 / 3, 4), h
    print(f"  accuracy 0.5 over n_evaluable={acc['n_evaluable']}, "
          f"skipped={acc['n_skipped_no_gt']}, CI[{acc['lo']}, {acc['hi']}]")


def test_option_parsing_handles_embedded_options():
    stem, opts = parse_options("What is to the right of the book?\nA. cup\nB. plant\nC. sofa")
    assert set(opts) == {"A", "B", "C"}, opts
    assert opts["A"] == "cup" and opts["C"] == "sofa", opts
    assert stem == "What is to the right of the book?", stem
    stem2, opts2 = parse_options("no options here at all")
    assert opts2 == {} and stem2 == "no options here at all", (stem2, opts2)
    print(f"  parsed {len(opts)} embedded options, and correctly found none in plain text")


def test_stub_is_reproducible_across_calls():
    """A randomised stub makes every reranking assertion depend on call order, not geometry."""
    d = write_scene()
    outs = []
    for _ in range(3):
        p = make_pipeline(StubMLLM(bias="A"), d)
        outs.append(_qa_result(p, Condition.SCENE_3D, True)["final_answer"])
    assert outs == [GT, GT, GT], outs
    print(f"  three independent runs agree: {outs}")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    print(f"running {len(ALL_TESTS)} offline pipeline/eval/analysis tests\n")
    failed = []
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as e:
            failed.append((fn.__name__, str(e)))
            print(f"  !! {fn.__name__} FAILED: {e}")
        except Exception as e:
            failed.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  !! {fn.__name__} ERROR: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} passed")
    for n, e in failed:
        print(f"  FAIL {n}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
