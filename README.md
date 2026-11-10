# PRISM-V: Programmable Reasoning with Integrated Spatial Modules with Retrieval-based Verification

Zero-shot spatial reasoning over an explorable 3D scene, with a retrieval-based geometric
verification layer that checks the model's spatial hypotheses before they become answers.

The perception stack (VGGT reconstruction, novel-view synthesis, camera-pose algebra,
MindCube evaluation) is inherited from the project's own prior pipeline and reused rather
than rewritten, so the baseline stays exactly reproducible and every number in the comparison
is attributable to the parts PRISM-V actually adds.

```
perceive                    verify                       reason
reconstruct 3D scene   ->   retrieve geometric     ->   execute the generated program,
lift masks into 3D          evidence, check spatial     answer; verification reranks
entities                    assertions                  the candidates before commit
(prismv.reconstruct_pipe,    (prismv.geover —           (prismv.agent,
 prismv.tools,                pure numpy)                prismv.pipeline)
 prismv.geover.grounding)
```

## Installation

```shell
git clone --recursive <your-fork> && cd PRISM-V
conda create -n prismv python=3.10 && conda activate prismv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # match your CUDA
pip3 install -r requirement.txt

export OPENAI_API_KEY=***      # code generation + answering
export GEMINI_API_KEY=***          # optional alternative backends
export OPEN_ROUTER_KEY=***       # optional alternative backends
```

`prismv.geover` is pure numpy — the verification layer, the metrics and the analysis module
import and run with **no** torch, open3d, trimesh, GPU or API key. That is what makes
`smoke` below work anywhere.

### Data and cached geometry

```shell
mkdir -p datasets && cd datasets
git clone https://github.com/mll-lab-nu/MindCube.git
cd MindCube && bash scripts/bash_scripts/download_data.bash && cd ../..

# reconstruct once, reuse across every ablation arm
python -m reconstruct_pipe --mode batch \
    --jsonl_path datasets/MindCube/data/raw/MindCube_tinybench.jsonl \
    --base_data_path datasets/MindCube/data \
    --gpu_ids 0,1
```

PRISM-V's `reconstruct_pipe` additionally writes `pointmaps.npz` per scene (the addressable
`(N,H,W,3)` pointmap plus depth confidence, colour and rescaled cameras). Without it a 2D
entity mask cannot be lifted into 3D, so scenes preprocessed before this stage existed will
ground nothing and verification will abstain — re-run the preprocessing to get it.

## Quick start

```shell
# 0. self-check, no GPU / no API key
python run_prismv.py smoke

# 1. audit trail for a single question
python run_prismv.py verify \
    --question "From the viewpoint presented in image 1, what is to the left of the book?" \
    --options A=cup B=plant C=sofa D=lamp \
    --processed_dir /tmp/prismv_offline_scene

# 2. the headline ablation: 3 input conditions x verification on/off
python run_prismv.py eval \
    --jsonl_path datasets/MindCube/data/raw/MindCube_tinybench.jsonl \
    --base_data_path datasets/MindCube/data \
    --processed_dir output/preprocessed \
    --tasks qa --all_conditions --ablate_geover \
    --max_entries 200 --save_results --output results/ablation.json

# 3. three tasks at once
python run_prismv.py eval --tasks qa localize describe ... # same flags

# 4. diagnostics over a finished run
python run_prismv.py analyze --report results/ablation.json \
    --output results/analysis.json
```

Tests: `python -m pytest tests/ -q`

## GeoVer

```python
from prismv.interface import PRISM_V, Scene

scene = Scene(["view0.jpg", "view1.jpg"], question, scene_id="abc")
PRISM_V.reconstruct(scene)                     # or load cached geometry
entities = PRISM_V.ground_entities(scene)      # masks -> 3D point sets

# one claim, graded and attributable
PRISM_V.verify(scene, relation="left_of", subject="plant", obj="book", ref_view=0)
# {'relation': 'left_of', 'status': 'supported', 'score': 0.9151,
#  'margin': 0.0682, 'detail': 'left_of @view0: angular offset +0.0682 rad',
#  'entities_used': ['plant', 'book']}

# a whole multiple-choice question, reranked
PRISM_V.check_options(scene, question, {"A": "cup", "B": "plant"}, ref_view=0)

# localisation: who best satisfies the relation
PRISM_V.rank_relations(scene, anchor="book", candidates=["cup", "plant"],
                       relation="right_of", ref_view=0)
```

**Verifiable relations.** Lateral and depth relations are evaluated in the reference
camera's frame; verticality along the gravity axis estimated from the cameras.

| relation | requires |
| --- | --- |
| `left_of` `right_of` `between` `occludes` `same_side` | a reference view |
| `closer_than` `farther_than` | a reference view (falls back to the scene centre) |
| `above` `below` | estimated gravity |
| `near` `far_from` `faces_toward` | entity centroids and scene scale only |

**Three-way verdicts.** `supported` / `contradicted` / `undetermined` are not two states.
`undetermined` covers a missing reference view, an entity reconstructed from too few points,
a vertical gap inside the bodies' own scale, and a relation outside the vocabulary. Mapping
those onto `contradicted` is what makes a verification layer score *below* the baseline it
checks, because sparse geometry would then veto correct answers.

**Abstention.** The reranker only overrides the model when it has hard evidence
(`min_support`) and its top choice is separated from the runner-up
(`abstain_threshold`). Otherwise it leaves the ranking alone and says so in `note`.

## Evaluation

`prismv.eval` reports, per cell of the matrix:

- `answer_accuracy` with bootstrap CIs; `condition_comparison` with **paired** bootstrap
  (the same scenes under each condition — unpaired intervals on a few hundred items
  overstate what is known)
- `verification_effect` — `fixes` (wrong→right) and `damages` (right→wrong) counted
  separately, plus `constraint_set_decisive_rate`: how often the geometry endorses exactly
  one option. That last number is the ceiling on reranking; on the ambiguous remainder the
  layer can only remove an impossible answer, never select the correct one
- `pipeline_health` — parse / execution / grounded / answered / evaluable rates, so accuracy
  is never averaged over the easy half of a benchmark by accident
- `localization_metrics` — acc@1/3/5 plus position error in scene-extent units
- `description_metrics` — geometric precision and contradiction rate of the spatial claims a
  generated description actually asserts
- `relation_level_accuracy`, `grounding_quality`

Datasets that carry only four-option QA ground truth (MindCube included) score only the `qa`
task. `task_coverage` prints which fields exist **before** any model runs, and an option
naming nothing the reconstruction grounded leaves `gt_entity` unset rather than guessing a
target — a task the adapter cannot be scored on is reported, not fabricated.

## Entity grounding

Three paths, in evidence-strength order, recorded in `provenance["grounding_path"]`:

1. `entity_points` — entities that already carry 3D points (`entities.json`), exact
2. `mask_lift` — 2D masks indexed into `pointmaps.npz`, with depth-confidence filtering
3. `cluster_fallback` — unsupervised 3D clusters labelled `cluster_0…`, deliberately
   unbindable by the hypothesis parser so unverifiable never becomes verified

`masks_without_pointmap` is reported when masks exist but the scene was preprocessed without
`pointmaps.npz`, instead of silently substituting a weaker path.

## Layout

```
prismv/
  interface.py       PRISM_V tool API: perception primitives + verification
  pipeline.py        perceive -> verify -> reason, input conditions, MLLM backends, tool recorder
  analysis.py        tool-strategy, reconstruction-quality, verification-mechanism sweeps
  geover/            evidence.py predicates.py grounding.py hypothesis.py verifier.py rerank.py
  eval/              datasets.py metrics.py runner.py
  agent/             template.py (PRISM-V prompts) code_query.py
reconstruct_pipe.py  VGGT preprocessing, now writing per-view pointmaps
tool/  agent/  base_model/vggt/   inherited (identifiers renamed only)
tests/test_geover_synthetic.py    18 offline tests, synthetic scene, known geometry
tests/test_pipeline_offline.py    23 offline tests, pipeline + eval + analysis
run_prismv.py        CLI: smoke / verify / eval / analyze
```

The inherited baseline modules (`baseline_interface.py`, `mindcube.py`, `agent/`, `tool/`)
are kept behaviourally intact and re-exported by `prismv` behind lazy imports. They were
mechanically renamed -- module name, facade class, the `exec` namespace key and the example
calls in the baseline prompt -- so the project carries one naming scheme. Nothing else about
the baseline's logic, prompts or control flow was altered; the rename is covered by
`tests/test_baseline_naming.py`, and previously generated programs from before the rename
should be re-generated rather than replayed.

## Baseline pipeline naming

The inherited baseline modules were renamed (`baseline_interface.py`, class `SpatialAPI`) so
the project carries a single naming scheme. The rename is mechanical: the class name, the
module name, the `exec` namespace key that generated programs resolve against, and every
example call in the baseline prompt were updated together, so the baseline arm still
generates, executes and scores exactly as before. No compatibility alias for the old name is
kept -- programs are always generated against the name the prompt teaches.

## Acknowledgement

Built on [VGGT](https://github.com/facebookresearch/vggt),
[CUT3R](https://github.com/CUT3R/CUT3R), [PySpatial](https://github.com/Zhanpeng1202/pySpatial), [MindCube](https://github.com/mll-lab-nu/MindCube)
and [VPGen](https://github.com/j-min/VPGen).

## Citation

```bibtex
@inproceedings{prismv2026,
  title     = {PRISM-V: Programmable Reasoning with Integrated Spatial Modules with
               Retrieval-based Verification},
  year      = {2026}
}
```
