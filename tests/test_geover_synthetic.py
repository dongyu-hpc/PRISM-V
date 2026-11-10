"""Offline unit tests for GeoVer (PRISM-V) on a synthetic scene with known geometry.

No GPU, no API key, no dataset: the scene is constructed directly so every expected
relation has a hand-computable ground truth. This is the harness that proves the
verification layer actually constrains answers rather than rubber-stamping them, and it
also pins the behaviours that are easy to regress -- reference-frame sensitivity in
particular, since "left of" flipping with the viewer is the whole reason the layer exists.

Run:  python tests/test_geover_synthetic.py     (also works under pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prismv.geover import (  # noqa: E402
    GeoVer,
    GeometricEvidenceIndex,
    SceneGraph,
    assertions_for_options,
    detect_reference_view,
    estimate_gravity,
    infer_pose_convention,
    match_entities,
)
from prismv.geover.evidence import EntityEvidence  # noqa: E402
from prismv.geover.predicates import (  # noqa: E402
    PREDICATES,
    camera_frame,
    relation_support_report,
)
from prismv.geover.rerank import AnswerReranker  # noqa: E402
from prismv.geover.verifier import GeoVerifier  # noqa: E402

RNG = np.random.RandomState(7)


def blob(center, radius=0.05, n=400):
    """A point cloud around ``center`` -- what mask-lifting produces for one entity."""
    return np.asarray(center, dtype=np.float64) + RNG.randn(n, 3) * radius


def view_identity():
    """Camera at the world origin looking down +Z, with +X right and +Y down."""
    return np.eye(4)


def view_turned_around():
    """Camera moved to (0,0,4) and yawed 180 deg: +X is now to the *left*."""
    R = np.diag([-1.0, 1.0, -1.0])
    origin = np.array([0.0, 0.0, 4.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -(R @ origin)
    return T


def build_graph():
    """Layout (camera 0 at origin, looking +Z; +X right, +Y down):

        lamp  (0, -0.5, 3.0)   far away, vertically above the others
        book  (-0.4, 0.1, 2.0) left of centre
        cup   (0.3, 0.1, 2.0)  right of centre
        plant (-0.6, 0.1, 2.2) further left than book
        sofa  (0.0, 0.2, 1.2)  nearest to the camera
    """
    # tests build this scene more than once and compare audit trails byte-for-byte, so the
    # point noise has to be reproducible rather than drawn from a running generator
    global RNG
    RNG = np.random.RandomState(7)
    specs = {
        "lamp": (0.0, -0.5, 3.0),
        "book": (-0.4, 0.1, 2.0),
        "cup": (0.3, 0.1, 2.0),
        "plant": (-0.6, 0.1, 2.2),
        "sofa": (0.0, 0.2, 1.2),
    }
    entities = []
    for name, c in specs.items():
        entities.append(EntityEvidence(name=name, points=blob(c), views=[0, 1],
                                       pixel_counts=[400, 400], confidence=1.0))
    extr = np.stack([view_identity(), view_turned_around()])
    pc = np.concatenate([e.points for e in entities], axis=0)
    return SceneGraph(entities=entities, extrinsics=extr,
                      gravity=estimate_gravity(extr), point_cloud=pc, scene_id="synthetic")


# ---------------------------------------------------------------------- sanity
def test_camera_frames_and_gravity():
    g = build_graph()
    f0 = camera_frame(g.extrinsics[0])
    assert np.allclose(f0["origin"], [0, 0, 0]), f0
    assert np.allclose(f0["forward"], [0, 0, 1])
    assert np.allclose(f0["right"], [1, 0, 0])
    f1 = camera_frame(g.extrinsics[1])
    assert np.allclose(f1["origin"], [0, 0, 4]), f1["origin"]
    assert np.allclose(f1["forward"], [0, 0, -1])
    assert np.allclose(f1["right"], [-1, 0, 0])
    # both cameras are upright, so the inferred gravity is +Y regardless of convention
    assert np.allclose(g.gravity, [0, 1, 0], atol=1e-8), g.gravity
    assert infer_pose_convention(g.extrinsics, g.point_cloud) is True
    print("  camera frames / gravity / pose convention OK")


def test_reference_view_detection():
    assert detect_reference_view("From the viewpoint presented in image 2, what is left?") == 1
    assert detect_reference_view("in image 1, what is to the right of the cup?") == 0
    assert detect_reference_view("what is next to the sofa?") is None
    assert detect_reference_view("based on the third image shown") == 2
    print("  reference-view detection OK")


# ---------------------------------------------------------------------- predicates
def test_directional_predicates_are_view_relative():
    """The central property: the same pair flips left/right when the viewer turns around."""
    g = build_graph()
    idx = GeometricEvidenceIndex(g)
    ver = GeoVerifier(idx)
    from prismv.geover.hypothesis import Assertion

    cup_right_of_book_v0 = Assertion("right_of", subject="cup", object="book", ref_view=0)
    cup_right_of_book_v1 = Assertion("right_of", subject="cup", object="book", ref_view=1)
    plant_left_of_book_v0 = Assertion("left_of", subject="plant", object="book", ref_view=0)

    c0 = ver.check(cup_right_of_book_v0)
    c1 = ver.check(cup_right_of_book_v1)
    p0 = ver.check(plant_left_of_book_v0)
    assert c0.status == "supported", c0.to_dict()
    assert c1.status == "contradicted", c1.to_dict()
    assert p0.status == "supported", p0.to_dict()
    # margins must have flipped sign, not merely changed magnitude
    assert c0.verdict.margin > 0 > c1.verdict.margin
    print(f"  cup right of book: view0 margin {c0.verdict.margin:+.3f} (supported), "
          f"view1 margin {c1.verdict.margin:+.3f} (contradicted)")


def test_depth_and_vertical_predicates():
    g = build_graph()
    idx = GeometricEvidenceIndex(g)
    ver = GeoVerifier(idx)
    from prismv.geover.hypothesis import Assertion

    checks = {
        "sofa closer than cup": ver.check(Assertion("closer_than", "sofa", "cup", ref_view=0)),
        "lamp farther than cup": ver.check(Assertion("farther_than", "lamp", "cup", ref_view=0)),
        "cup closer than lamp": ver.check(Assertion("closer_than", "cup", "lamp", ref_view=0)),
        "lamp above sofa": ver.check(Assertion("above", "lamp", "sofa", ref_view=0)),
        "sofa below lamp": ver.check(Assertion("below", "sofa", "lamp", ref_view=0)),
        "book above lamp": ver.check(Assertion("above", "book", "lamp", ref_view=0)),
    }
    assert checks["sofa closer than cup"].status == "supported"
    assert checks["lamp farther than cup"].status == "supported"
    assert checks["cup closer than lamp"].status == "supported"
    assert checks["lamp above sofa"].status == "supported", checks["lamp above sofa"].to_dict()
    assert checks["sofa below lamp"].status == "supported"
    assert checks["book above lamp"].status == "contradicted"
    print("  distance ordering + verticality OK")


def test_missing_reference_view_is_undetermined():
    """A lateral claim with no viewer must abstain, never guess off the world frame."""
    g = build_graph()
    ver = GeoVerifier(GeometricEvidenceIndex(g))
    from prismv.geover.hypothesis import Assertion
    c = ver.check(Assertion("right_of", subject="cup", object="book", ref_view=None))
    assert c.status == "undetermined", c.to_dict()
    assert c.verdict.undetermined
    print("  lateral relation without reference view -> undetermined OK")


def test_sparse_entity_is_not_trusted():
    """5 lifted pixels should not become geometric authority."""
    g = build_graph()
    g.entities.append(EntityEvidence(name="vase", points=RNG.randn(6, 3) * 0.01,
                                     views=[0], pixel_counts=[6], confidence=0.1))
    ver = GeoVerifier(GeometricEvidenceIndex(g))
    from prismv.geover.hypothesis import Assertion
    c = ver.check(Assertion("right_of", subject="vase", object="book", ref_view=0))
    assert c.status == "undetermined", c.to_dict()
    assert "points" in c.verdict.detail
    print("  sparse entity -> undetermined OK")


def test_between_and_occlusion():
    g = build_graph()
    # something laterally mid-way between book(-0.4) and cup(0.3) at similar depth
    g.entities.append(EntityEvidence(name="rug", points=blob((0.0, 0.15, 2.05)), views=[0],
                                     pixel_counts=[400], confidence=1.0))
    # a wide panel, close to the camera, genuinely in the line of sight of the cup
    g.entities.append(EntityEvidence(name="panel", points=blob((0.15, 0.1, 1.5), radius=0.35, n=800),
                                     views=[0], pixel_counts=[800], confidence=1.0))
    ver = GeoVerifier(GeometricEvidenceIndex(g))
    from prismv.geover.hypothesis import Assertion
    b = ver.check(Assertion("between", subject="rug", object="book", third="cup", ref_view=0))
    assert b.status == "supported", b.to_dict()
    nb = ver.check(Assertion("between", subject="cup", object="book", third="rug", ref_view=0))
    assert nb.status == "contradicted", nb.to_dict()

    # Occlusion needs BOTH angular overlap and depth ordering. The sofa is nearer than the
    # cup but only 5 cm across, so it subtends far less than the 0.19 rad separating them:
    # asserting it blocks the cup is exactly the kind of geometric non-sequitur the layer
    # exists to catch, so it must come back contradicted rather than "close enough".
    small = ver.check(Assertion("occludes", subject="sofa", object="cup", ref_view=0))
    assert small.status == "contradicted", small.to_dict()
    wide = ver.check(Assertion("occludes", subject="panel", object="cup", ref_view=0))
    assert wide.status == "supported", wide.to_dict()
    # and the reverse ordering (occluded object in front) must not count either
    reverse = ver.check(Assertion("occludes", subject="cup", object="panel", ref_view=0))
    assert reverse.status == "contradicted", reverse.to_dict()
    print(f"  between OK; occlusion gated on overlap+depth "
          f"(small/near rejected, wide/nearer accepted, depth-reversed rejected)")



# ---------------------------------------------------------------------- hypothesis
def test_entity_matching_is_constrained_to_the_graph():
    g = build_graph()
    idx = GeometricEvidenceIndex(g)
    found = [n for _, _, n in match_entities("Is the cup to the right of the book?", idx)]
    assert "cup" in found and "book" in found, found
    # a hallucinated entity must NOT bind onto a real one
    assert match_entities("the aquarium", idx) == [], match_entities("the aquarium", idx)
    assert match_entities("the sneakers", idx) == [], "no footwear in this scene"
    print("  constrained entity matching OK")


def test_mcq_compiles_to_per_option_assertions():
    g = build_graph()
    idx = GeometricEvidenceIndex(g)
    q = "From the viewpoint presented in image 1, what is to the right of the book?"
    opts = {"A": "cup", "B": "plant", "C": "sofa", "D": "a giraffe"}
    parsed = assertions_for_options(q, opts, idx)
    assert parsed.ref_view == 0
    letters = {a.source for a in parsed.option_assertions}
    assert letters == {"option:A", "option:B", "option:C"}, letters
    # option D names nothing in the reconstruction -> no assertion -> treated as unverifiable
    assert not any(a.source == "option:D" for a in parsed.option_assertions)
    a0 = parsed.option_assertions[0]
    assert a0.relation == "right_of" and a0.object == "book" and a0.subject == "cup", a0.to_dict()
    print(f"  MCQ compiled to {len(parsed.option_assertions)} assertions, "
          f"unverifiable option D produced none")


# ---------------------------------------------------------------------- rerank
def test_rerank_overrides_a_wrong_answer():
    """MLLM says plant (wrong); geometry supports cup -> verification must flip it."""
    gv = GeoVer(graph=build_graph(), verification_weight=1.0, min_support=1)
    q = "From the viewpoint presented in image 1, what is to the right of the book?"
    opts = {"A": "cup", "B": "plant", "C": "sofa", "D": "lamp"}
    out = gv.verify_and_rerank(q, opts, vlm_answer="B",
                               vlm_confidence=[2.0, 3.0, 0.0, 0.0])
    assert out.applied, out.rerank.note
    assert out.answer == "A", out.rerank.to_dict()
    print(f"  rerank flipped B -> A ({out.rerank.note}), "
          f"coverage {out.report.option_coverage()}")


def test_rerank_respects_the_reference_view():
    """Same question, camera turned around: the geometrically endorsed option must change."""
    q = "From the viewpoint presented in image 2, what is to the right of the book?"
    opts = {"A": "cup", "B": "plant", "C": "sofa", "D": "lamp"}
    gv = GeoVer(graph=build_graph(), verification_weight=1.0, min_support=1)
    out = gv.verify_and_rerank(q, opts, vlm_answer="A", vlm_confidence=[3.0, 2.0, 0.0, 0.0])
    assert out.parsed.ref_view == 1
    # cup is contradicted from view 2; plant is the entity now on the right
    cov = out.report.option_coverage()
    assert cov["A"]["contradicted"] >= 1, cov
    assert cov["B"]["supported"] >= 1, cov
    assert out.answer == "B", (out.rerank.to_dict(), cov)
    print(f"  with viewer turned around, rerank moved A -> B; coverage {cov}")


def test_rerank_abstains_without_evidence():
    """No grounded entities -> the layer must leave the MLLM's answer untouched."""
    empty = SceneGraph(entities=[], extrinsics=np.eye(4).reshape(1, 4, 4), gravity=None)
    gv = GeoVer(graph=empty)
    out = gv.verify_and_rerank("what is to the right of the book?",
                              {"A": "cup", "B": "plant"}, vlm_answer="B")
    assert out.applied is False
    assert out.answer == "B"
    print(f"  abstention OK ({out.rerank.note})")


def test_rerank_weight_zero_is_the_geoVer_ablation():
    """weight=0 keeps the report but must not move the answer: the -GeoVer arm."""
    gv = GeoVer(graph=build_graph(), verification_weight=0.0)
    q = "From the viewpoint presented in image 1, what is to the right of the book?"
    opts = {"A": "cup", "B": "plant", "C": "sofa", "D": "lamp"}
    out = gv.verify_and_rerank(q, opts, vlm_answer="B",
                               vlm_confidence=[2.0, 3.0, 0.0, 0.0])
    assert out.report.n_checks > 0, "verification still runs and reports"
    assert out.answer == "B", "but with lambda=0 the prior stands"
    print("  weight=0 ablation preserves the original answer OK")


def test_unsupported_relation_is_never_asserted():
    """An assertion in a vocabulary GeoVer has no predicate for must not be scored."""
    gv = GeoVer(graph=build_graph(), enabled_relations=["right_of"])
    from prismv.geover.hypothesis import Assertion
    assert gv.verifier.check(Assertion("smells_like", "cup", "book", ref_view=0)) is None
    assert gv.verifier.check(Assertion("right_of", "cup", "book", ref_view=0)).status == "supported"
    print("  relation allow-list OK; coverage table:")
    for rel, req in relation_support_report():
        assert rel in PREDICATES
        print(f"      {rel:<14} requires {req}")


# ---------------------------------------------------------------------- determinism
def test_verification_is_deterministic():
    gv = GeoVer(graph=build_graph())
    q = "From the viewpoint presented in image 1, what is to the right of the book?"
    opts = {"A": "cup", "B": "plant", "C": "sofa", "D": "lamp"}
    r1 = gv.verify_and_rerank(q, opts, vlm_answer="B").to_dict()
    r2 = gv.verify_and_rerank(q, opts, vlm_answer="B").to_dict()
    assert r1 == r2, "identical inputs must produce identical audit trails"
    # every check must name the evidence that decided it
    assert r1["verification"]["counts"]["supported"] > 0
    for chk in r1["verification"]["checks"]:
        assert chk["entities_used"], chk
        assert chk["detail"], chk
    print("  deterministic + fully attributable checks OK")


def test_layer_defers_when_geometry_endorses_several_options():
    """The behaviour that keeps verification additive rather than a rival reasoner.

    cup, sofa and lamp are ALL to the right of the book, so the constraint set genuinely
    does not single out one answer. Under a flat prior the correct move is to abstain and
    leave the MLLM's ranking alone -- if the layer manufactured a winner here it would be
    claiming discriminative power its evidence does not have.
    """
    gv = GeoVer(graph=build_graph())
    q = "From the viewpoint presented in image 1, what is to the right of the book?"
    opts = {"A": "cup", "B": "plant", "C": "sofa", "D": "lamp"}
    out = gv.verify_and_rerank(q, opts, vlm_answer="B")
    cov = out.report.option_coverage()
    endorsed = [l for l, c in cov.items() if c["supported"] > 0]
    assert len(endorsed) >= 2, cov
    assert out.applied is False, out.rerank.to_dict()
    assert out.answer == "B"
    assert out.rerank.margin < 0.02, out.rerank.margin
    print(f"  {endorsed} all endorsed by geometry -> abstained, kept the model's B")


def test_a_prior_ending_the_tie_lets_verification_decide():
    """Same under-determined constraint set, but two options already split by the prior:
    now removing the contradicted one is a real, evidence-backed correction."""
    gv = GeoVer(graph=build_graph())
    q = "From the viewpoint presented in image 1, what is to the right of the book?"
    opts = {"A": "cup", "B": "plant"}
    out = gv.verify_and_rerank(q, opts, vlm_answer="B", vlm_confidence=[1.0, 1.2])
    assert out.applied and out.answer == "A", out.rerank.to_dict()
    print("  contradicted option demoted under a near-tie prior OK")



def test_report_is_serialisable():
    import json

    gv = GeoVer(graph=build_graph())
    out = gv.verify_and_rerank(
        "From the viewpoint presented in image 1, is the lamp above the sofa?",
        {"A": "yes", "B": "no"}, vlm_answer="B")
    s = json.dumps(out.to_dict(), indent=2)
    assert "margin" in s and "provenance" in s
    print(f"  JSON audit trail serialises OK ({len(s)} bytes)")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    print(f"running {len(ALL_TESTS)} GeoVer tests on a synthetic scene\n")
    failed = []
    for fn in ALL_TESTS:
        name = fn.__name__
        try:
            fn()
        except AssertionError as e:
            failed.append((name, e))
            print(f"  !! {name} FAILED: {e}")
        except Exception as e:
            failed.append((name, e))
            print(f"  !! {name} ERROR: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} passed")
    if failed:
        for n, e in failed:
            print(f"  FAIL {n}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
