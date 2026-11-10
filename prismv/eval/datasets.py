"""Dataset adapters for PRISM-V evaluation.

MindCube is the benchmark the inherited pipeline was built for and it is a four-option
multiple-choice format only. PRISM-V claims three tasks, so this module normalises whatever
a benchmark actually provides into a common task record, and is explicit about which task
inputs are *available* versus *synthesised*. Silently inventing ground truth for the
description and localisation tasks is how a multi-task evaluation ends up measuring its own
adapter.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

# MindCube tags its scenes by directory: .../among/..., .../around/..., .../rotation/...
_SCENE_TYPES = ("among", "around", "rotation")


def scene_type_of(images: Iterable[str]) -> str:
    for p in images or []:
        low = str(p).lower()
        for t in _SCENE_TYPES:
            if t in low:
                return t
    return "unknown"


def parse_options(question: str) -> Tuple[str, Dict[str, str]]:
    """Split a MindCube-style stem from its lettered options.

    Options may be embedded in the question string or supplied as a separate field; both
    occur in released spatial benchmarks, so the adapter handles either and the stem is
    returned with option text stripped (a stem that still contains its options leaks the
    answer space into prompts and inflates apparent model competence).
    """
    text = str(question or "")
    opts: Dict[str, str] = {}
    pat = re.compile(r"(?:^|\n|\s)([A-D])[.)\s]\s*(.+?)(?=(?:\n|\s+[A-D][.)\s])|$)",
                     re.DOTALL)
    for m in pat.finditer(text):
        letter = m.group(1).upper()
        if letter in opts:
            continue
        opts[letter] = re.sub(r"\s+", " ", m.group(2)).strip()
    if not opts:
        return text.strip(), {}
    stem = pat.split(text)[0].strip()
    if not stem:
        stem = text.strip()
    return stem, opts


def normalize_option_answer(value: Any) -> str:
    s = str(value or "").strip().upper()
    m = re.match(r"\s*([A-D])\b", s)
    return m.group(1) if m else s


def load_jsonl(path: str, max_entries: Optional[int] = None) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_entries and len(rows) >= max_entries:
                break
    return rows


def to_task_record(entry: Dict, base_data_path: Optional[str] = None,
                   task: str = "qa") -> Dict[str, Any]:
    """Convert one benchmark entry into PRISM-V's task record.

    ``task`` selects which fields are required and whether ground truth exists:

    ``qa``           needs options + gt_answer. Directly supported by MindCube.
    ``localize``     needs a target entity. Taken from ``gt_entity`` when the benchmark
                     supplies one; otherwise resolved from the gt option's text against the
                     scene graph. If neither works, ``gt_entity`` stays None and the
                     localisation metric simply does not score that item -- it is not
                     guessed.
    ``describe``     needs relation ground truth. Only scored when ``gt_relations`` is
                     present; the geometric-precision half (is what the model said
                     consistent with the reconstruction) still works without it, and the
                     two halves are reported separately by the metrics module.
    """
    images = [str(p) for p in (entry.get("images") or [])]
    if base_data_path:
        images = [p if os.path.isabs(p) else os.path.join(base_data_path, p) for p in images]
    raw_q = entry.get("question") or entry.get("prompt") or ""
    stem, opts = parse_options(raw_q)
    if entry.get("options"):
        o = entry["options"]
        if isinstance(o, dict):
            opts = {str(k).upper(): str(v) for k, v in o.items()}
        elif isinstance(o, list):
            opts = {chr(65 + i): str(v) for i, v in enumerate(o)}

    rec: Dict[str, Any] = {
        "scene_id": entry.get("id"),
        "images": images,
        "question": stem,
        "full_question": raw_q,
        "options": opts,
        "gt_answer": normalize_option_answer(entry.get("gt_answer", entry.get("answer", ""))),
        "scene_type": entry.get("scene_type") or scene_type_of(images),
        "task": task,
        "gt_entity": entry.get("gt_entity"),
        "gt_position": entry.get("gt_position"),
        "gt_relations": entry.get("gt_relations"),
        "entity_masks": entry.get("entity_masks") or entry.get("masks"),
    }

    if task == "localize" and not rec["gt_entity"]:
        # resolve the correct option's text into an entity name via simple normalisation;
        # unmatched option text leaves gt_entity None rather than becoming a false target
        letter = rec["gt_answer"]
        text = (opts.get(letter) or "").lower()
        names = [str(n).lower() for n in _mention_names(entry)]
        for n in names:
            if n and re.search(rf"\b{re.escape(n)}\b", text):
                rec["gt_entity"] = n
                break
    return rec


def _mention_names(entry: Dict) -> List[str]:
    """Entity names the benchmark already knows about, if it lists any."""
    names: List[str] = []
    for spec in (entry.get("entity_masks") or entry.get("masks") or []):
        if isinstance(spec, dict) and spec.get("name"):
            names.append(str(spec["name"]))
    for k in ("objects", "entities", "gt_objects"):
        for v in (entry.get(k) or []):
            names.append(str(v.get("name") if isinstance(v, dict) else v))
    return names


def task_coverage(records: List[Dict]) -> Dict[str, Any]:
    """Report what each task can actually be scored on -- before anyone runs a model.

    A multi-task benchmark claim that cannot say how many items carry relation ground
    truth is a claim about the adapter, not the dataset.
    """
    out: Dict[str, Any] = {}
    for task in ("qa", "localize", "describe"):
        sub = [r for r in records if r["task"] == task] or records
        out[task] = {
            "n": len(sub),
            "with_options": sum(1 for r in sub if r.get("options")),
            "with_gt_answer": sum(1 for r in sub if r.get("gt_answer")),
            "with_gt_entity": sum(1 for r in sub if r.get("gt_entity")),
            "with_gt_relations": sum(1 for r in sub if r.get("gt_relations")),
            "with_entity_masks": sum(1 for r in sub if r.get("entity_masks")),
        }
    out["scene_types"] = dict(Counter(r.get("scene_type") for r in records))
    return out


def load_mindcube(jsonl_path: str, base_data_path: Optional[str] = None,
                  max_entries: Optional[int] = None, task: str = "qa") -> List[Dict]:
    """Convenience loader for the MindCube benchmark used by the inherited pipeline."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"MindCube JSONL not found at {jsonl_path}. Fetch it with: "
            f"git clone https://github.com/mll-lab-nu/MindCube datasets/MindCube && "
            f"(cd datasets/MindCube && bash scripts/bash_scripts/download_data.bash)")
    return [to_task_record(e, base_data_path, task=task)
            for e in load_jsonl(jsonl_path, max_entries)]
