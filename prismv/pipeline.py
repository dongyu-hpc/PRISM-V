"""The PRISM-V perceive -> verify -> reason pipeline.

The inherited loop was: write a program, run it, hand whatever it rendered to a VLM, take the
answer. Where the inherited loop stops -- perceive, render, ask -- PRISM-V inserts a constraint check
between "the model believes something" and "that belief becomes the answer", and makes the
input condition an explicit variable so the framework's contribution can be attributed rather
than asserted.

Everything here is dependency-injected: the MLLM behind a thin client protocol with an
offline stub implementation, so the orchestration, the ablation matrix and the metrics can
all be exercised on a machine with no GPU, no weights and no API key. That is not a
convenience -- a pipeline whose control flow can only be tested end-to-end against a paid
API is a pipeline whose control flow is never tested.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from prismv.interface import PRISM_V, Reconstruction, Scene
from prismv.geover import GeoVer
from prismv.geover.evidence import GeometricEvidenceIndex, SceneGraph
from prismv.geover.hypothesis import assertions_from_text

# ------------------------------------------------------------------ input conditions
class Condition:
    """What the model is allowed to see. One axis of the headline ablation.

    ``TEXT_ONLY`` and ``IMAGE_2D`` are not strawmen: they are the two things a spatial
    question gets in every non-tool-using deployment, and the gap between them and
    ``SCENE_3D`` is the gap PRISM-V claims to close.
    """

    TEXT_ONLY = "text_only"      # question + option strings, no pixels, no 3D
    IMAGE_2D = "image_2d"        # multi-view images straight to the MLLM
    SCENE_3D = "scene_3d"        # tool-built explorable 3D scene

    ALL = (TEXT_ONLY, IMAGE_2D, SCENE_3D)

    @staticmethod
    def uses_program(c: str) -> bool:
        return c == Condition.SCENE_3D


# ------------------------------------------------------------------ answer object
@dataclass
class AnswerResult:
    """One model answer, with whatever confidence signal the backend exposed."""

    answer: str                       # option letter
    reasoning: str = ""
    confidence: Optional[List[float]] = None   # per-option logprobs, aligned to ``letters``
    letters: List[str] = field(default_factory=list)
    raw: Any = None
    latency_s: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict:
        return {"answer": self.answer, "reasoning": self.reasoning,
                "letters": self.letters,
                "confidence": None if self.confidence is None
                else [round(float(c), 4) for c in self.confidence],
                "latency_s": round(self.latency_s, 3), "error": self.error}


class MLLMClient(Protocol):
    """The only interface the pipeline needs from a vision-language model."""

    name: str

    def answer_question(self, question: str, options: Dict[str, str],
                        images: Optional[Sequence[str]] = None,
                        visual_clue: Any = None,
                        system: str = "", ground_with_3d: bool = False) -> AnswerResult:
        ...


@dataclass
class ToolCall:
    """A single tool invocation by an executed program -- the unit the tool-use analysis
    aggregates over."""

    tool: str
    args: Dict = field(default_factory=dict)
    ok: bool = True
    latency_ms: int = 0
    note: str = ""

    def to_dict(self) -> Dict:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "args": self.args, "note": self.note}


class ToolRecorder:
    """Wraps the tool API so every call an executed program makes is observable.

    Without this, "the model chose good tool strategies" and "the model got lucky" produce
    identical final answers. Recording the call sequence is what makes the tool-calling
    analysis in the paper's fourth contribution something other than a vibe.
    """

    def __init__(self):
        self.calls: List[ToolCall] = []

    def record(self, tool: str, ok: bool = True, latency_ms: int = 0,
               args: Optional[Dict] = None, note: str = "") -> ToolCall:
        c = ToolCall(tool=tool, ok=ok, latency_ms=latency_ms, args=args or {}, note=note)
        self.calls.append(c)
        return c

    def summary(self) -> Dict:
        names = [c.tool for c in self.calls]
        return {
            "n_calls": len(self.calls),
            "unique_tools": sorted(set(names)),
            "n_unique_tools": len(set(names)),
            "failure_rate": round(sum(1 for c in self.calls if not c.ok) / max(1, len(names)), 4),
            "sequence": names,
            "total_latency_ms": sum(c.latency_ms for c in self.calls),
        }


# ------------------------------------------------------------------ offline stub
class StubMLLM:
    """Deterministic offline stand-in.

    It answers from the question text and whatever clue it was handed, with no network. Not
    a spatial reasoner -- its purpose is to make the pipeline's *control flow* testable:
    reranking, fallback, condition routing, logging and metric computation all run against
    it exactly as they do against a real backend.
    """

    name = "stub"

    def __init__(self, bias: str = "first", seed: int = 0,
                 scripted_text: Optional[str] = None):
        self.bias = bias
        self.seed = seed
        # ``scripted_text`` lets a test hand the describe arm real spatial claims, so the
        # claim-extraction -> verification path can be exercised without a live model.
        self.scripted_text = scripted_text
        self.calls: List[Dict] = []

    def answer_question(self, question: str, options: Dict[str, str],
                        images: Optional[Sequence[str]] = None, visual_clue: Any = None,
                        system: str = "", ground_with_3d: bool = False) -> AnswerResult:
        t0 = time.perf_counter()
        letters = list(options.keys())
        self.calls.append({"question": question, "n_images": len(images or []),
                           "has_clue": visual_clue is not None,
                           "ground_with_3d": ground_with_3d})
        if not letters:
            # free-form tasks (scene description) have no option set; the stub still has to
            # return text, otherwise the describe arm cannot be tested offline at all
            return AnswerResult(answer="", letters=[],
                                reasoning=self.scripted_text or "",
                                latency_s=time.perf_counter() - t0)
        if self.bias == "first":
            pick = letters[0]
        elif self.bias == "random":
            rng = np.random.RandomState(self.seed + len(self.calls))
            pick = letters[int(rng.randint(len(letters)))]
        else:
            pick = self.bias if self.bias in letters else letters[0]
        # Deterministic on purpose. A randomised prior makes every reranking assertion in
        # the suite depend on call order rather than on the geometry under test -- the usual
        # origin of a "verification test that fails on Tuesdays". The chosen option sits one
        # unit above the rest so the prior is non-degenerate but fully reproducible.
        conf = [0.0] * len(letters)
        conf[letters.index(pick)] = 1.0
        return AnswerResult(answer=pick, reasoning=f"[stub] chose {pick} for: {question[:60]}",
                            confidence=conf, letters=letters,
                            latency_s=time.perf_counter() - t0)


# ------------------------------------------------------------------ real backend
class OpenAIMLLM:
    """The gpt-5 backend used by the inherited agent, kept behind the same protocol.

    Imported lazily so that installing PRISM-V for verification-only work does not require
    the openai package.
    """

    name = "gpt-5"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5",
                 reasoning_effort: str = "low"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.reasoning_effort = reasoning_effort
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set; pass api_key or use StubMLLM")
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, timeout=180)

    def answer_question(self, question: str, options: Dict[str, str],
                        images: Optional[Sequence[str]] = None, visual_clue: Any = None,
                        system: str = "", ground_with_3d: bool = False) -> AnswerResult:
        import base64

        from pydantic import BaseModel, Field

        letters = list(options.keys())

        class _Answer(BaseModel):
            reasoning: str
            answer: str = Field(description="one option letter")

        opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        content: List[Dict[str, Any]] = [{
            "type": "input_text",
            "text": (system + "\n\n") if system else "",
        }]
        content[0]["text"] += f"Question: {question}\nOptions:\n{opt_text}\n"
        if ground_with_3d:
            content[0]["text"] += ("A 3D scene was reconstructed from these images and a "
                                   "program was executed against it. Reason from the "
                                   "provided clue.\n")
        for path in (images or []):
            if os.path.exists(path):
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                ext = os.path.splitext(path)[1].lower().lstrip(".")
                mime = {"png": "image/png", "jpg": "image/jpeg",
                        "jpeg": "image/jpeg"}.get(ext, "image/png")
                content.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
        if visual_clue is not None:
            from prismv.agent.answer import o3d_image_to_data_url  # lazy, needs open3d

            clues = visual_clue if isinstance(visual_clue, list) else [visual_clue]
            for clue in clues:
                if isinstance(clue, str):
                    content.append({"type": "input_text", "text": f"Visual clue: {clue}"})
                else:
                    content.append({"type": "input_image",
                                    "image_url": o3d_image_to_data_url(clue)})

        t0 = time.perf_counter()
        try:
            resp = self.client.responses.parse(
                model=self.model,
                input=[{"role": "system", "content": "You answer spatial reasoning questions."},
                       {"role": "user", "content": content}],
                max_output_tokens=2000, text_format=_Answer)
            parsed = resp.output_parsed
            return AnswerResult(answer=str(parsed.answer).strip()[:1].upper(),
                                reasoning=parsed.reasoning, letters=letters,
                                raw=parsed, latency_s=time.perf_counter() - t0)
        except Exception as e:
            return AnswerResult(answer="", letters=letters, error=str(e),
                                latency_s=time.perf_counter() - t0)


# ------------------------------------------------------------------ code execution
def parse_code(response: str) -> str:
    """Pull the program out of a model response.

    Fenced ```python``` blocks are what the prompt asks for and what the inherited agent
    assumed. Returning an unfenced-but-valid program when one is handed over directly is
    not sloppiness: generators differ (some emit bare code, some emit prose plus a fence),
    and silently producing an empty program turns a formatting mismatch into
    "the model failed to solve this scene", which is a measurement error rather than a
    result. The fallback still requires the ``def program`` contract, so prose never
    accidentally executes.
    """
    if not response:
        return ""
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if m and "def program" in m.group(1):
        return m.group(1).strip()
    if "def program" in response:
        # A generator that emits bare code instead of a fenced block is a formatting
        # difference, not a failed attempt. Returning an empty string here would record it
        # as parse_ok=False and let the whole scene fall through to the QA fallback, which
        # quietly measures something other than what ran. The `def program` contract is
        # still required, so prose cannot execute by accident.
        return response[response.index("def program"):].replace("```", "").rstrip()
    return ""


def build_execution_namespace(recorder: Optional[ToolRecorder] = None) -> Dict[str, Any]:
    """The globals an LLM-written program runs against.

    When ``recorder`` is supplied the API surface is wrapped so each call is logged. The
    wrapper has to preserve the original signature and docstring, because the code generator
    is shown the same API specification and its output is validated against those names.
    """
    import prismv.interface as iface

    # Only the documented names are exposed to generated programs.
    # legacy alias stays so previously emitted programs keep resolving.
    ns = {"PRISM_V": iface.PRISM_V, "PRISMV": iface.PRISM_V,
          "Scene": iface.Scene, "Reconstruction": iface.Reconstruction,
          "np": np, "numpy": np}
    if recorder is None:
        return ns

    class _Recording:
        def __init__(self, inner, rec: ToolRecorder):
            self._inner, self._rec = inner, rec

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if not callable(attr) or name.startswith("_"):
                return attr

            def wrapper(*a, **kw):
                t0 = time.perf_counter()
                try:
                    out = attr(*a, **kw)
                except Exception as e:
                    self._rec.record(name, ok=False,
                                     latency_ms=int((time.perf_counter() - t0) * 1000),
                                     note=f"{type(e).__name__}: {e}")
                    raise
                self._rec.record(name, ok=True,
                                 latency_ms=int((time.perf_counter() - t0) * 1000),
                                 args={k: _brief(v) for k, v in kw.items()})
                return out

            wrapper.__name__ = name
            wrapper.__doc__ = getattr(attr, "__doc__", None)
            return wrapper

    ns["PRISM_V"] = _Recording(iface.PRISM_V, recorder)
    ns["PRISMV"] = ns["PRISM_V"]
    return ns


def _brief(v: Any) -> Any:
    """Cheap, log-safe summary of an argument (shapes, not contents)."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape}"
    if hasattr(v, "shape"):
        return f"{type(v).__name__}{getattr(v, 'shape', '')}"
    return type(v).__name__


def execute_program(code: str, scene: Scene, recorder: Optional[ToolRecorder] = None,
                    timeout_s: float = 0.0):
    """Run an LLM-written program against a scene and return its visual clue.

    ``timeout_s`` bounds execution: generated code is arbitrary and can loop, so one bad
    scene must not stall a sweep of hundreds. 0 disables the limit.
    """
    import signal

    ns = build_execution_namespace(recorder)
    namespace: Dict[str, Any] = {}
    src = code
    try:
        src = code.encode().decode("unicode_escape")
    except Exception:
        pass  # already-decoded code is the common case; unicode_escape would mangle it

    if timeout_s and timeout_s > 0 and hasattr(signal, "SIGALRM"):
        def _fire(signum, frame):
            raise TimeoutError(f"program exceeded {timeout_s}s")

        prev = signal.signal(signal.SIGALRM, _fire)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            exec(src, ns, namespace)
            program = namespace["program"]
            out = program(scene)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)
        return out
    exec(src, ns, namespace)
    program = namespace["program"]
    return program(scene)


# ------------------------------------------------------------------ the pipeline
@dataclass
class PipelineResult:
    """Everything one question produces, stage by stage."""

    scene_id: Optional[str]
    question: str
    options: Dict[str, str]
    condition: str
    used_geover: bool
    expected_answer: Optional[str] = None

    code: str = ""
    parse_ok: bool = False
    execution_ok: bool = False
    clue_type: str = ""
    mllm_answer: Optional[str] = None
    final_answer: Optional[str] = None
    correct: Optional[bool] = None
    reasoning: str = ""
    tool_summary: Dict = field(default_factory=dict)
    verification: Dict = field(default_factory=dict)
    provenance: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "scene_id": self.scene_id, "question": self.question, "options": self.options,
            "condition": self.condition, "used_geover": self.used_geover,
            "expected_answer": self.expected_answer, "code": self.code,
            "parse_ok": self.parse_ok, "execution_ok": self.execution_ok,
            "clue_type": self.clue_type, "mllm_answer": self.mllm_answer,
            "final_answer": self.final_answer, "correct": self.correct,
            "reasoning": self.reasoning, "tool_summary": self.tool_summary,
            "verification": self.verification, "provenance": self.provenance,
            "errors": self.errors, "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


class PRISMVPipeline:
    """Perceive -> verify -> reason, with the ablation switches exposed as arguments."""

    def __init__(self, mllm: MLLMClient, code_generator: Optional[Callable[[Scene], str]] = None,
                 processed_dir: Optional[str] = None, entity_masks: Optional[Sequence[Dict]] = None,
                 verification_weight: float = 1.0, min_support: int = 1,
                 conf_thres: float = 0.0, exec_timeout: float = 180.0,
                 fallback_clusters: int = 0):
        self.mllm = mllm
        self.code_generator = code_generator
        self.entity_masks = entity_masks
        self.verification_weight = verification_weight
        self.min_support = min_support
        self.conf_thres = conf_thres
        self.exec_timeout = exec_timeout
        self.fallback_clusters = fallback_clusters
        PRISM_V.PROCESSED_BASE_DIR = processed_dir

    # ---------------------------------------------------------------- code writing
    def _generate_program(self, scene: Scene) -> str:
        if self.code_generator is not None:
            resp = self.code_generator(scene)
            return parse_code(resp if isinstance(resp, str) else "")
        # Offline default: ground the scene and enumerate what was grounded. It deliberately
        # does NOT call reconstruct() -- ground_entities already prefers cached geometry, so
        # adding the call would make every default-arm program fail outright on a machine
        # without the perception stack, and the run would report execution_rate 0 while the
        # pipeline quietly recovered through its own grounding fallback.
        return ('def program(input_scene):\n'
                '    entities = PRISM_V.ground_entities(input_scene)\n'
                '    if len(entities) == 0:\n'
                '        return "no entities grounded"\n'
                '    return "grounded: " + ", ".join(entities.names)\n')

    # ------------------------------------------------------------------- main
    def run(self, question: str, options: Dict[str, str],
            images: Optional[Sequence[str]] = None, condition: str = Condition.SCENE_3D,
            used_geover: bool = True, scene_id: Optional[str] = None,
            expected_answer: Optional[str] = None,
            entity_masks: Optional[Sequence[Dict]] = None) -> PipelineResult:
        res = PipelineResult(scene_id=scene_id, question=question, options=options,
                             condition=condition, used_geover=used_geover,
                             expected_answer=expected_answer)
        scene = Scene(list(images or []), question, scene_id=scene_id)
        recorder = ToolRecorder()

        # ---------------- perceive ----------------
        if Condition.uses_program(condition):
            t0 = time.perf_counter()
            res.code = self._generate_program(scene)
            res.parse_ok = bool(res.code.strip())
            if not res.parse_ok:
                res.errors.append("no ```python``` block in generated code")
            else:
                try:
                    clue = execute_program(res.code, scene, recorder, self.exec_timeout)
                    res.execution_ok = clue is not None
                    res.clue_type = type(clue).__name__
                except Exception as e:
                    res.errors.append(f"execution: {type(e).__name__}: {e}")
                    clue = None
            res.timings["perceive"] = time.perf_counter() - t0
            res.tool_summary = recorder.summary()
            if scene.reconstruction is None and scene.images:
                try:
                    PRISM_V.reconstruct(scene)
                except Exception as e:
                    res.errors.append(f"reconstruct: {e}")
            if scene.entities is None:
                try:
                    PRISM_V.ground_entities(scene, entity_masks=entity_masks or self.entity_masks,
                                            conf_thres=self.conf_thres,
                                            fallback_clusters=self.fallback_clusters)
                except Exception as e:
                    res.errors.append(f"grounding: {e}")
            res.provenance = dict(getattr(scene, "entity_provenance", {}))
            visual_clue = clue if res.execution_ok else None
        elif condition == Condition.IMAGE_2D:
            visual_clue = None
        else:  # TEXT_ONLY: the model gets the question string and nothing else
            images = []
            visual_clue = None

        # ---------------- reason (first pass) ----------------
        t0 = time.perf_counter()
        ans = self.mllm.answer_question(
            question, options, images=images, visual_clue=visual_clue,
            ground_with_3d=Condition.uses_program(condition))
        res.mllm_answer = ans.answer or None
        res.reasoning = ans.reasoning
        if ans.error:
            res.errors.append(f"mllm: {ans.error}")
        res.timings["reason"] = time.perf_counter() - t0

        # ---------------- verify ----------------
        final = ans.answer
        if used_geover and Condition.uses_program(condition):
            t0 = time.perf_counter()
            graph = getattr(scene, "entities", None)
            if graph is not None and len(graph.entities):
                gv = GeoVer(graph=graph, verification_weight=self.verification_weight,
                            min_support=self.min_support)
                parsed = gv.parse(question, options, reasoning=res.reasoning)
                letters = list(options.keys())
                report = gv.verify(parsed, letters)
                rerank = gv.reranker.score(report, letters, prior=ans.confidence,
                                           original_answer=ans.answer)
                final = rerank.answer
                res.verification = {
                    "applied": rerank.applied, "note": rerank.note,
                    # echo the reference frame the layer inferred. Every lateral verdict in
                    # the report is conditioned on it, so an auditor reading a flipped answer
                    # has to be able to see which viewer the geometry was judged from -- and
                    # that it was inferred rather than assumed silently.
                    "ref_view_inferred": parsed.ref_view,
                    "n_assertions": len(parsed.option_assertions)
                    + len(parsed.question_assertions)
                    + len(parsed.reasoning_assertions),
                    "counts": report.counts(),
                    "option_coverage": report.option_coverage(),
                    "ranked": [r.to_dict() for r in rerank.ranked],
                    "checks": [c.to_dict() for c in report.checks],
                    "hypotheses": parsed.summary(),
                    "changed_answer": bool(rerank.applied and final != ans.answer),
                }
            else:
                res.verification = {"applied": False, "note": "no grounded entities",
                                    "counts": {"supported": 0, "contradicted": 0,
                                               "undetermined": 0},
                                    "changed_answer": False}
            res.timings["verify"] = time.perf_counter() - t0

        res.final_answer = final or None
        if expected_answer and res.final_answer:
            res.correct = _normalize(res.final_answer) == _normalize(expected_answer)
        elif res.final_answer and expected_answer is None:
            res.correct = None
        return res


def _normalize(answer: Any) -> str:
    """Reduce an answer to its option letter; datasets decorate gt_answer inconsistently."""
    if answer is None:
        return ""
    text = str(answer).strip().upper()
    m = re.match(r"\s*([A-Z])\b", text)
    return m.group(1) if m else text
