"""Hypothesis parsing for GeoVer (PRISM-V).

Turning free-form model output into *checkable* assertions is the interface between the
reasoning side of PRISM-V and the geometry side. It is deliberately the narrowest part of
the pipeline: extraction is constrained to the entity names that the reconstruction
actually grounds, and to the relation vocabulary the predicate library can evaluate.

That constraint is the point. Open-information-extraction would happily produce
"the lamp is left of the plant" for a scene where the reconstruction has no plant, and
verification would then have nothing to check against. Constrained extraction converts an
unverifiable sentence into an explicitly *unresolved* one, which the reranker treats as
 abstention instead of as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from prismv.geover.predicates import RELATION_ALIASES

# Triggers are tried longest-first so "to the right of" wins over "right of".
TRIGGERS: List[Tuple[str, str]] = sorted(
    [(alias, relation) for relation, aliases in RELATION_ALIASES.items() for alias in aliases],
    key=lambda kv: -len(kv[0]),
)

_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "there", "it", "its", "is", "are",
    "was", "were", "be", "been", "being", "and", "or", "of", "to", "in", "on", "at", "by",
    "with", "from", "into", "onto", "what", "which", "who", "where", "when", "how", "why",
    "do", "does", "did", "would", "could", "should", "will", "can", "may", "might",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "if", "then", "than", "so", "as", "but", "not", "no",
    "yes", "about", "above", "under", "over", "same", "both", "either", "neither",
    "image", "images", "view", "views", "picture", "pictures", "photo", "photos",
    "following", "shown", "based", "question", "option", "options", "answer", "closest",
}


@dataclass
class Assertion:
    """One spatial claim that GeoVer can try to check.

    ``source`` records where the claim came from ("option:A", "reasoning", "description")
    so the report can show which candidate answer each piece of geometric evidence bears on.
    """

    relation: str
    subject: Optional[str] = None
    object: Optional[str] = None
    third: Optional[str] = None
    ref_view: Optional[int] = None
    source: str = "question"
    text: str = ""
    weight: float = 1.0
    meta: Dict = field(default_factory=dict)

    def entities(self) -> List[str]:
        return [e for e in (self.subject, self.object, self.third) if e]

    def is_complete(self) -> bool:
        return bool(self.subject and self.object)

    def to_dict(self) -> Dict:
        return {"relation": self.relation, "subject": self.subject, "object": self.object,
                "third": self.third, "ref_view": self.ref_view, "source": self.source,
                "text": self.text, "weight": self.weight}


@dataclass
class ParsedHypotheses:
    """Everything extracted from one question, kept separate by origin."""

    question_assertions: List[Assertion] = field(default_factory=list)
    option_assertions: List[Assertion] = field(default_factory=list)
    reasoning_assertions: List[Assertion] = field(default_factory=list)
    ref_view: Optional[int] = None
    unresolved_triggers: List[str] = field(default_factory=list)

    def all_for_option(self, letter: str) -> List[Assertion]:
        return [a for a in self.option_assertions if a.source == f"option:{letter}"]

    def summary(self) -> Dict:
        return {
            "n_question": len(self.question_assertions),
            "n_option": len(self.option_assertions),
            "n_reasoning": len(self.reasoning_assertions),
            "ref_view": self.ref_view,
            "n_unresolved_triggers": len(self.unresolved_triggers),
        }


# ------------------------------------------------------------------ entity matching
def _norm(text) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    return re.sub(r"\s+", " ", s).strip()


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _norm_tokens(text: str) -> List[str]:
    return [_singular(w) for w in _norm(text).split()]


def match_entities(text: str, index) -> List[Tuple[int, int, str]]:
    """Find spans of *reconstruction-grounded* entities inside free text.

    Returns (start, end, entity_name) sorted by start. Substring matching against the
    scene graph's names and aliases is used rather than generic noun-phrase extraction:
    the set of entities GeoVer can verify is exactly the set the grounding step produced,
    and a mention outside that set cannot become evidence no matter how well it parses.
    """
    low = _norm(text)
    if not low or index is None:
        return []
    candidates = []
    for e in index.graph.entities:
        for name in [e.name] + list(e.aliases):
            n = _norm(name)
            if len(n) < 3:
                continue
            start = 0
            while True:
                i = low.find(n, start)
                if i < 0:
                    break
                j = i + len(n)
                left_ok = (i == 0) or low[i - 1] == " "
                right_ok = (j == len(low)) or low[j] == " "
                if left_ok and right_ok:
                    candidates.append((i, j, e.name))
                start = i + 1
            # single-token alias tolerant of plural ("sneakers" -> "sneaker")
            words = n.split()
            if len(words) == 1:
                for m in re.finditer(r"\b([a-z0-9]+)\b", low):
                    if _singular(m.group(1)) == _singular(words[0]):
                        candidates.append((m.start(), m.end(), e.name))
    if not candidates:
        return []
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    chosen: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []
    for s, e, name in candidates:
        if any(not (e <= os_ or s >= oe) for os_, oe in occupied):
            continue
        chosen.append((s, e, name))
        occupied.append((s, e))
    return chosen


def detect_relations(text: str) -> List[Tuple[int, int, str]]:
    """Locate spatial-relation triggers in text (longest match wins, no overlap)."""
    low = _norm(text)
    found: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []
    for trig, relation in TRIGGERS:
        t = _norm(trig)
        if not t:
            continue
        start = 0
        while True:
            i = low.find(t, start)
            if i < 0:
                break
            j = i + len(t)
            left_ok = (i == 0) or low[i - 1] == " "
            right_ok = (j == len(low)) or low[j] == " "
            if left_ok and right_ok and not any(not (j <= os_ or i >= oe) for os_, oe in occupied):
                found.append((i, j, relation))
                occupied.append((i, j))
            start = i + 1
    found.sort(key=lambda f: f[0])
    return found


# ------------------------------------------------------------------ reference view
_VIEW_PATTERNS = [
    r"(?:image|view|picture|photo)\s*(?:number\s*)?([1-9])\b",
    r"(?:first|1st)\s+view", r"(?:second|2nd)\s+view", r"(?:third|3rd)\s+view",
    r"(?:fourth|4th)\s+view",
]
_WORD_NUM = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}


def detect_reference_view(text: str) -> Optional[int]:
    """Which view the question's left/right/front language is anchored to.

    Returns a 0-based index (matching ``extrinsics[i]``). If the question never anchors to
    a specific view, this returns None and view-relative predicates are reported as
    undetermined -- which is a real limitation of the input, not of the geometry.
    """
    low = _norm(text)
    if not low:
        return None
    # "from the viewpoint presented in image 2" / "as shown in image 1" / "in view 3"
    m = re.search(r"(?:from|in|as shown in|presented in|shown in)?\s*"
                  r"(?:the\s+)?(?:viewpoint|view|image|picture|photo)\s*"
                  r"(?:presented |shown )?(?:in )?(?:image |view )?([1-9])\b", low)
    if m:
        return int(m.group(1)) - 1
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b"
                  r"\s+(?:image|view|picture|photo)", low)
    if m:
        return _WORD_NUM[m.group(1)] - 1
    return None


# ------------------------------------------------------------------ clause building
def _entity_after(entities, pos):
    for s, e, name in entities:
        if s >= pos:
            return name
    return None


def _entity_before(entities, pos):
    best = None
    for s, e, name in entities:
        if e <= pos:
            best = name
    return best


def extract_clauses(text: str, index, source: str = "question",
                    ref_view: Optional[int] = None) -> List[Assertion]:
    """Pair each relation trigger with its nearest subject (before) and object (after).

    A trigger with no grounded object before it is kept with ``object=None``: that is the
    multiple-choice case ("what is to the right of the sneaker?" -- the answer option
    *is* the subject), and it is resolved later by :func:`assertions_for_options`.
    """
    entities = match_entities(text, index)
    triggers = detect_relations(text)
    out: List[Assertion] = []
    for i, j, relation in triggers:
        subj = _entity_before(entities, i)
        obj = _entity_after(entities, j)
        third = None
        if relation == "between" and obj:
            # "between A and B": the second named entity after the object is the third slot
            tail_entities = [(s, e, n) for (s, e, n) in entities if s > j and n != obj]
            third = tail_entities[0][2] if tail_entities else None
        if subj and obj and subj == obj:
            subj = None
        out.append(Assertion(relation=relation, subject=subj, object=obj, third=third,
                             ref_view=ref_view, source=source, text=text,
                             meta={"trigger_pos": i}))
    return out


def assertions_for_options(question: str, options: Dict[str, str], index,
                           parsed: Optional[ParsedHypotheses] = None,
                           max_entities_per_option: int = 3) -> ParsedHypotheses:
    """Compile a multiple-choice question into per-option checkable assertions.

    For each option, entity mentions inside the option text are bound as the *subject* of
    any open slot in the question stem, giving one assertion per (option, candidate
    subject, stem relation) triple. Options that name no grounded entity produce no
    assertions, so the reranker treats them as unverifiable rather than as contradicted.
    """
    parsed = parsed or ParsedHypotheses()
    stem_view = detect_reference_view(question)
    if parsed.ref_view is None:
        parsed.ref_view = stem_view
    clauses = extract_clauses(question, index, source="question", ref_view=stem_view)
    open_slots = [c for c in clauses if c.object and not c.subject]
    closed = [c for c in clauses if c.is_complete()]
    parsed.question_assertions.extend(closed)
    for c in clauses:
        if not c.is_complete() and not c.object:
            parsed.unresolved_triggers.append(c.relation)

    for letter, text in (options or {}).items():
        mentions = match_entities(text, index)
        names = []
        for _, _, n in mentions:
            if n not in names:
                names.append(n)
        if not names:
            # an option may itself be a full clause ("the TV is to the right of the sofa")
            inner = extract_clauses(text, index, source=f"option:{letter}", ref_view=stem_view)
            for a in inner:
                if a.is_complete():
                    a.weight = 0.9
                    parsed.option_assertions.append(a)
            continue
        for slot in open_slots:
            for name in names[:max_entities_per_option]:
                parsed.option_assertions.append(Assertion(
                    relation=slot.relation,
                    subject=name,
                    object=slot.object,
                    third=slot.third,
                    ref_view=stem_view if slot.ref_view is None else slot.ref_view,
                    source=f"option:{letter}",
                    text=f"{text} | {question}",
                    weight=1.0,
                    meta={"option_text": text, "n_option_mentions": len(names)},
                ))
    return parsed


def assertions_from_text(text: str, index, source: str = "reasoning",
                         ref_view: Optional[int] = None) -> List[Assertion]:
    """Extract claims from a reasoning trace or a generated scene description.

    Used for consistency scoring: a chain of thought that asserts "the lamp is left of the
    sofa" can be checked against the same geometry that decided the answer, and an
    unsupported claim in the justification is itself a signal about answer reliability.
    """
    if not text:
        return []
    view = ref_view if ref_view is not None else detect_reference_view(text)
    out = []
    for chunk in re.split(r"(?<=[.?;])\s+", str(text)):
        if not chunk.strip():
            continue
        for a in extract_clauses(chunk, index, source=source, ref_view=view):
            if a.is_complete():
                a.weight = 0.6 if source == "reasoning" else 0.8
                out.append(a)
    return out
