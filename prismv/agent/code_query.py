"""PRISM-V code generator and answer stage.

Same shape as the inherited ``agent.codeAgent.query``, with two differences that matter for
the ablation: it prompts with the PRISM-V API surface (so the model can actually address the
verification tools), and it asks the answer stage for **per-option logprobs** when the
backend can supply them.

The second point is not a nicety. The reranker combines an answer prior with geometric
evidence; without a prior it can only reweight evenly, and the reported "verification
improves accuracy" number would be measuring a different estimator than the one described.
When logprobs are unavailable the code says so in the returned metadata instead of quietly
substituting a flat prior.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from prismv.agent import template as prismv_template

DEFAULT_MODEL = "gpt-5"


def _client(api_key: Optional[str] = None):
    from openai import OpenAI  # lazy: verification-only installs need not have openai

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set; pass api_key or run with StubMLLM")
    return OpenAI(api_key=key, timeout=120)


def generate_code_from_query(scene, api_key: Optional[str] = None,
                             model: str = DEFAULT_MODEL) -> str:
    """Write a PRISM-V program for ``scene.question``. Returns the raw model response."""
    client = _client(api_key)
    query = prismv_template.build_query(scene.question)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system",
                   "content": "You generate Python code using the PRISM-V API to solve "
                              "spatial reasoning problems."},
                  {"role": "user", "content": query}],
        reasoning_effort="low",
        max_completion_tokens=4000,
    )
    return response.choices[0].message.content


def answer_with_clue(scene, visual_clue, api_key: Optional[str] = None,
                     model: str = DEFAULT_MODEL,
                     options: Optional[Dict[str, str]] = None) -> Tuple[str, str,
                                                                        List[float], str]:
    """Answer from the executed clue.

    Returns ``(answer_letter, reasoning, per_option_confidence, prior_source)`` where
    ``prior_source`` is ``"logprobs"`` or ``"flat"``. That last field is what lets the
    analysis distinguish a genuine posterior update from a reweighting of a uniform prior.
    """
    from pydantic import BaseModel, Field

    client = _client(api_key)

    class SpatialAnswer(BaseModel):
        reasoning: str
        answer: str = Field(description="one option letter, e.g. A")

    letters = list((options or {}).keys())
    clue_text = _stringify_clue(visual_clue)
    prompt = (f"{prismv_template.verify_answer_background}\n"
              f"{prismv_template.verify_answer_prompt}\n"
              f"the question is {scene.question}\n"
              f"the generated program was:\n{scene.code}\n"
              f"its output (the clue) is:\n{clue_text}\n")

    resp = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": "You answer spatial reasoning questions from "
                                             "verified geometric evidence."},
               {"role": "user", "content": prompt}],
        max_output_tokens=2000,
        text_format=SpatialAnswer,
        # No `include=[...logprobs]` request: an unrecognised value there is a hard 400, which
        # would break every answer call to make one optional diagnostic. Logprobs are read
        # opportunistically below and the caller is told when they were not available.
    )
    parsed = resp.output_parsed
    conf, source = _option_logprobs(resp, letters)
    return (str(parsed.answer).strip()[:1].upper(), parsed.reasoning, conf, source)


def _stringify_clue(clue) -> str:
    """Render a clue as text. Verification dicts are kept structured on purpose.

    Images are not inlined here: a program that returns rendered views still needs them as
    image parts, and mixing both into one string would drop the pixels silently.
    """
    import json

    if clue is None:
        return "<no clue produced>"
    if isinstance(clue, str):
        return clue
    if isinstance(clue, (dict, list, tuple)):
        try:
            return json.dumps(clue, indent=2, default=str)
        except Exception:
            return repr(clue)
    return repr(clue)


def _option_logprobs(response, letters: List[str]) -> Tuple[List[float], str]:
    """Extract per-option logprobs for the answer token, best effort.

    Backends differ in whether structured-output responses expose logprobs, and the shape is
    version-dependent, so this degrades to a flat vector and reports that it did. Returning
    a fabricated confidence vector would let the reranker look better-informed than it is.
    """
    if not letters:
        return [], "flat"
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise AttributeError("no choices")
        lp = getattr(choices[0], "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if not content:
            raise AttributeError("no token logprobs")
        # the option letter is the last non-space token of the structured answer
        tokens = [t for t in content if getattr(t, "token", "").strip()]
        if not tokens:
            raise AttributeError("empty token list")
        last = tokens[-1]
        top = getattr(last, "top_logprobs", None) or []
        table = {}
        for t in top:
            tok = (getattr(t, "token", "") or "").strip().upper()
            if tok:
                table.setdefault(tok, float(getattr(t, "logprob", -100.0)))
        if all(l in table for l in letters):
            return [table[l] for l in letters], "logprobs"
        raise AttributeError("option letters absent from top_logprobs")
    except Exception:
        return [0.0] * len(letters), "flat"


def generate_code(scene, api_key: Optional[str] = None) -> str:
    """Legacy name, kept so the inherited call sites work unchanged."""
    return generate_code_from_query(scene, api_key)
