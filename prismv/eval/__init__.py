"""PRISM-V evaluation: task drivers, dataset adapters, metrics, ablation matrix."""

from prismv.eval import datasets, metrics, runner  # noqa: F401
from prismv.eval.datasets import (  # noqa: F401
    load_jsonl,
    load_mindcube,
    normalize_option_answer,
    parse_options,
    scene_type_of,
    task_coverage,
    to_task_record,
)
from prismv.eval.metrics import (  # noqa: F401
    answer_accuracy,
    bootstrap_ci,
    condition_comparison,
    description_metrics,
    grounding_quality,
    localization_metrics,
    paired_diff,
    pipeline_health,
    relation_level_accuracy,
    summarize,
    verification_effect,
)
from prismv.eval.runner import Cell, EvalRunner  # noqa: F401

__all__ = ["datasets", "metrics", "runner", "EvalRunner", "Cell", "load_mindcube",
           "load_jsonl", "to_task_record", "parse_options", "normalize_option_answer",
           "scene_type_of", "task_coverage", "answer_accuracy", "verification_effect",
           "relation_level_accuracy", "grounding_quality", "localization_metrics",
           "description_metrics", "pipeline_health", "condition_comparison", "summarize",
           "bootstrap_ci", "paired_diff"]
