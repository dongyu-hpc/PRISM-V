"""PRISM-V prompt templates for the code-generating agent.

Deliberately a *separate* module from ``agent/prompt/template.py``, so the baseline arm keeps
its own prompt untouched apart from the mechanical identifier rename applied tree-wide. If
both arms shared one prompt file, the comparison would silently measure a prompt change
instead of the framework change it claims to measure.

The extension below is the API surface PRISM-V adds: entity grounding and retrieval-based
geometric verification. Showing these to the generator is what turns GeoVer from a
post-hoc audit into something a program can *ask* -- "is the cup actually to the right of
the book from view 1" -- and get a graded, attributable answer for.
"""

from __future__ import annotations

# keep the baseline text importable from here too, so a single import site suffices for both
from agent.prompt.template import (  # noqa: F401
    answer_background,
    answer_prompt,
    code_generation_prompt as baseline_code_generation_prompt,
    example_problems as baseline_example_problems,
    task_description as baseline_task_description,
    without_visual_clue_background,
)

# --------------------------------------------------------------------- API specification
api_specification = """
    This is the PRISM-V API. It provides 3D reconstruction tools *and* geometric
    verification tools. A Scene holds the input images and the question.

    class Reconstruction:
        # point_cloud : numpy array of 3D points
        # extrinsics  : per-view camera matrices, shape (N, 4, 4) or (N, 3, 4);
        #               extrinsics[i] is the world->camera pose of view i
        #               (0-based: image 1 -> index 0)
        # intrinsics  : camera intrinsics, may be None

    class Scene:
        "Holds the input images, the question, and whatever perception has produced."
        question : str
        images   : list of image paths
        reconstruction : Reconstruction or None
        entities : the grounded scene graph or None

    class PRISM_V:
        "Interface for 3D vision and verification tools."

        # ---------------- perception (inherited) ----------------
        @staticmethod
        def reconstruct(scene: Scene) -> Reconstruction:
            "Reconstruct the scene from its images."

        @staticmethod
        def describe_camera_motion(recon: Reconstruction) -> str:
            "Text description of the camera trajectory across views."

        @staticmethod
        def synthesize_novel_view(recon, new_camera_pose, width=512, height=512, out_path=None):
            "Render from a new viewpoint. Omit out_path to get an image object back."

        @staticmethod
        def rotate_right(extrinsic, angle=None, recon=None): ...
        @staticmethod
        def rotate_left(extrinsic, angle=None, recon=None): ...
        @staticmethod
        def move_forward(extrinsic, distance=None): ...
        @staticmethod
        def move_backward(extrinsic, distance=None): ...
        @staticmethod
        def turn_around(extrinsic, recon=None): ...

        # ---------------- grounding (PRISM-V) ----------------
        @staticmethod
        def ground_entities(scene: Scene, entity_masks=None, conf_thres=0.0,
                            fallback_clusters=0) -> EntitySet:
            "Lift 2D entity masks into 3D point sets. Call this before any verify call.
            Returns an EntitySet; len(entities) may be 0, and entities.names lists labels."

        @staticmethod
        def list_entities(scene: Scene) -> list:
            "Names of the entities that were actually grounded. Never invent a name:
            only entities in this list can be verified."

        @staticmethod
        def entity_descriptions(scene: Scene) -> list:
            "Per-entity evidence: point counts, contributing views, confidence."

        @staticmethod
        def scene_statistics(scene: Scene) -> dict:
            "Entity count, view count, scene extent, whether gravity was estimated."

        # ---------------- verification (PRISM-V / GeoVer) ----------------
        @staticmethod
        def verify(scene, relation, subject, obj, third=None, ref_view=None) -> dict:
            "Check ONE spatial claim against the reconstructed geometry.
            ref_view is REQUIRED for left_of/right_of/between/occludes/same_side:
            pass the 0-based index of the image whose viewpoint defines 'left'/'right'.
            Returns {'status': 'supported'|'contradicted'|'undetermined',
                    'score': float in [0,1], 'margin': signed float, 'detail': str,
                    'entities_used': [...]}.
            'undetermined' means the evidence does not license an answer -- it is NOT
            the same as 'contradicted', so do not treat it as a refutation."

        @staticmethod
        def verify_many(scene, claims: list) -> list:
            "Batch verify. Each claim is a dict with keys relation, subject, object,
            third, ref_view."

        @staticmethod
        def check_options(scene, question, options: dict, ref_view=None,
                          verification_weight=1.0, min_support=1) -> dict:
            "Verify every option against the geometry and rerank them.
            Returns {'answer', 'applied', 'note', 'counts', 'option_coverage',
                     'ranked': [...], 'checks': [...], 'ref_view_inferred': int|None}."

        @staticmethod
        def rank_relations(scene, anchor, candidates, relation, ref_view=None) -> list:
            "Rank candidate entities by how strongly the geometry endorses
            'candidate <relation> anchor'. Note the anchor is the REFERENT, not the
            thing being located. Returns a list sorted best-first."

        @staticmethod
        def verifiable_relations() -> list:
            "[[relation, requirement], ...] -- the relations GeoVer can check and what
            each one consumes. Ask this before relying on an unusual relation."

        Verifiable relations:
            left_of, right_of          (need ref_view)
            closer_than, farther_than  (need ref_view, fall back to the scene centre)
            above, below               (need an estimated gravity axis)
            near, far_from             (centroids + scene scale only)
            between                    (needs ref_view plus a third entity)
            occludes                   (needs ref_view; depth AND angular overlap)
            faces_toward, same_side

    Use verification instead of guessing whenever the question turns on a spatial
    relation that the reconstruction can settle. Prefer it over rendering an image when
    the answer is a relation between two objects you can name.
"""

task_description = """
    You are asked to solve a spatial reasoning problem. The input is one or more images
    plus a natural language question designed to test spatial reasoning, which is hard to
    answer directly as a vision language model. To help, you have access to the following
    Python API, which can both *construct* an explorable 3D scene from the images and
    *check* spatial claims against it.
"""

example_problems = """
    Problem 1 -- relation between two named objects, settled by geometry:
    Question: "Based on these three images of the same shelf: from the viewpoint of image 1,
    what is to the left of the blue notebook?
    A. the desk lamp
    B. the coffee mug
    C. the plant
    D. the keyboard"

    How to solve: this is a relation between two things the reconstruction can ground, so
    verify it rather than eyeballing a render. Ground the entities first, then let the
    geometry rank the candidates. Pass ref_view=0 because image 1 defines the viewpoint.

    ```python
    def program(input_scene: Scene):
        PRISM_V.reconstruct(input_scene)
        entities = PRISM_V.ground_entities(input_scene)
        if len(entities) == 0:
            return "no entities could be grounded, fall back to rendering"
        ranked = PRISM_V.rank_relations(
            input_scene, anchor="blue notebook",
            candidates=["desk lamp", "coffee mug", "plant", "keyboard"],
            relation="left_of", ref_view=0)
        return {"ranked": ranked, "grounded": entities.names}
    ```
    The ranked list with its margins is the visual clue; the best-supported candidate is
    the answer, and 'undetermined' entries must not be read as refutations.

    Problem 2 -- check the model's own hypothesis before committing:
    Question: "In image 2, is the monitor closer to the camera than the printer?
    A. Yes  B. No"

    ```python
    def program(input_scene: Scene):
        PRISM_V.reconstruct(input_scene)
        PRISM_V.ground_entities(input_scene)
        claim = PRISM_V.verify(input_scene, relation="closer_than",
                               subject="monitor", obj="printer", ref_view=1)
        return claim
    ```
    Read 'status': supported -> A, contradicted -> B, undetermined -> answer from the images.

    Problem 3 -- when a relation cannot be settled, render instead:
    Question: "From image 1, what appears in the doorway behind the sofa?
    A. a staircase  B. a mirror  C. a window  D. nothing"

    How to solve: this asks what is visible from a viewpoint, which the entity graph cannot
    settle. Render the direction instead -- verification is the wrong tool for an appearance
    question.

    ```python
    def program(input_scene: Scene):
        reconstruction3D = PRISM_V.reconstruct(input_scene)
        novel_viewpoint = PRISM_V.rotate_right(reconstruction3D.extrinsics[0],
                                               recon=reconstruction3D)
        return PRISM_V.synthesize_novel_view(reconstruction3D, novel_viewpoint)
    ```

    Problem 4 -- trajectory questions need no verification:
    Question: "In which direction did I move between the first and second view?
    A. forward-left  B. right  C. left  D. forward-right"

    ```python
    def program(input_scene: Scene):
        reconstruction3D = PRISM_V.reconstruct(input_scene)
        return PRISM_V.describe_camera_motion(reconstruction3D)
    ```
"""

code_generation_prompt = """
    Now use the PRISM-V API to write a Python function that solves the problem.
    You may reason first, but the final code must be a single compact ```python ``` block.
    The function must be named program and take one Scene argument:
    ```python
    def program(input_scene: Scene):
        ...
        return ...
    ```
    Rules:
    1. Call PRISM_V.list_entities or use the return value of ground_entities before relying
       on an entity name -- a name that was not grounded cannot be verified.
    2. For left/right/between/occlusion claims, always pass ref_view. If the question never
       states a viewpoint, do not invent one; render instead.
    3. Return the verification dicts (they carry the margin and the deciding detail) so the
       answer can be audited, rather than reducing them to a boolean yourself.
    4. When no API directly answers the question, render one or two images that best help
       solve it and return them as the visual clue.
"""

# Prompt used for the answer stage of the PRISM-V arm.
verify_answer_background = """
    We are solving a spatial reasoning problem that is hard to answer directly as a vision
    language model. A Python program was generated against the PRISM-V API and executed,
    producing a clue that may include geometric verification results:

        status = 'supported'     the reconstruction endorses the claim
        status = 'contradicted'  the reconstruction rules the claim out
        status = 'undetermined'  the evidence does not license either judgement

    Treat 'undetermined' as absence of evidence, never as a refutation. When the clue
    carries 'margin' and 'detail', weight your answer by them rather than by your own
    impression of the layout, and prefer the option with the strongest geometric support
    among those not contradicted.
"""

verify_answer_prompt = """
    Using the executed program's clue, answer the question. Return your reasoning and the
    single best option letter.
"""


def build_query(question: str) -> str:
    """Assemble the full generation prompt for one question."""
    return (f"{task_description}\n{api_specification}\n{example_problems}\n"
            f"{code_generation_prompt}\nthe question is {question}")
