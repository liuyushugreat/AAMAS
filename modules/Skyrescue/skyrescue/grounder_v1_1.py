"""Deterministic v1.1 semantic admission gate used by AAMAS Exp1.

The gate is label-isolated at inference time.  It only sees the model's
candidate, the scenario card, and the operator instruction.  Development
aliases are optional and are never required for an online decision.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping

from .workflow import CompilationResult, compile_generated_candidate


GENERIC_TARGETS = {
    "", "那里", "那儿", "那边", "那片区域", "该区域", "目标区域", "指定区域",
    "灾区", "现场", "附近", "外围", "unknown", "unspecified", "待确认",
}
UNRESOLVED_MARKERS = ("未指定", "未知地点", "地点不明", "待确认", "unspecified", "needs_expert_grounding")


def normalize_v11(value: Any) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "村子": "村落", "山里": "山区", "上方": "上空", "水面": "水域",
        "周边": "外围", "附近": "外围", "那个": "", "那处": "", "那条": "",
        "那片": "", "一处": "", "已确认": "", "确认好": "", "确认的": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _similarity(target: str, context: str) -> float:
    if not target or not context:
        return 0.0
    if target in context:
        return 1.0
    return round(SequenceMatcher(None, target, context).ratio(), 6)


@dataclass(frozen=True)
class GrounderV11Anchor:
    target_text: str
    normalized_target: str
    resolved: bool
    anchor_ids: tuple[str, ...]
    confidence: float
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ground_target_v11(
    target_text: str,
    scenario_card: str,
    instruction_text: str,
    *,
    threshold: float = 0.80,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> GrounderV11Anchor:
    """Resolve a candidate target against observable context only.

    Exact contextual mentions are preferred.  A fuzzy match is accepted only
    when it is sufficiently similar to one context clause and is not generic.
    ``aliases`` may contain Dev-only aliases, but no Gold field is required at
    inference time and no HeldOut label is consulted.
    """

    raw = str(target_text or "")
    target = normalize_v11(raw)
    if any(marker in raw.lower() for marker in UNRESOLVED_MARKERS):
        return GrounderV11Anchor(raw, target, False, (), 1.0, "marker", "explicit_unresolved")
    if target in {normalize_v11(value) for value in GENERIC_TARGETS} or len(target) < 2:
        return GrounderV11Anchor(raw, target, False, (), 0.0, "none", "generic_or_empty")

    context_parts = [scenario_card, instruction_text]
    clauses = [normalize_v11(part) for text in context_parts for part in re.split(r"[，。；、：:！？!?\n—]+", text)]
    clauses = [clause for clause in clauses if len(clause) >= 2]
    exact = [clause for clause in clauses if target in clause]
    if exact:
        anchor_key = hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
        return GrounderV11Anchor(raw, target, True, (f"ctx:{anchor_key}",), 1.0, "context_exact", "target_mentioned")

    alias_hits: list[tuple[str, str]] = []
    for canonical_id, values in (aliases or {}).items():
        normalized_values = {normalize_v11(value) for value in values}
        if target in normalized_values:
            alias_hits.append((canonical_id, target))
    if len(alias_hits) == 1:
        return GrounderV11Anchor(raw, target, True, (alias_hits[0][0],), 0.99, "dev_alias", "unique_dev_alias")
    if len(alias_hits) > 1:
        return GrounderV11Anchor(raw, target, False, (), 0.99, "dev_alias", "ambiguous_dev_alias")

    scored = sorted(((_similarity(target, clause), clause) for clause in clauses), reverse=True)
    if not scored:
        return GrounderV11Anchor(raw, target, False, (), 0.0, "none", "no_context")
    score, _ = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if score >= threshold and score - second >= 0.04:
        anchor_key = hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
        return GrounderV11Anchor(raw, target, True, (f"ctx:{anchor_key}",), score, "context_fuzzy", "unique_fuzzy_context")
    return GrounderV11Anchor(raw, target, False, (), score, "context_fuzzy", "low_or_ambiguous_similarity")


@dataclass(frozen=True)
class GroundedV11Compilation:
    compilation: CompilationResult
    anchor: GrounderV11Anchor

    def to_dict(self) -> dict[str, Any]:
        return {"compilation": self.compilation.to_dict(), "anchor": self.anchor.to_dict()}


def compile_grounded_candidate_v11(
    candidate: dict[str, Any],
    scenario_card: str,
    instruction_text: str,
    *,
    threshold: float = 0.80,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> GroundedV11Compilation:
    base = compile_generated_candidate(candidate)
    anchor = ground_target_v11(
        str(candidate.get("target_zone", "")), scenario_card, instruction_text,
        threshold=threshold, aliases=aliases,
    )
    if not base.schema_valid or not base.executable or anchor.resolved:
        if base.executable and anchor.resolved:
            tasks = [dict(task, target_anchor_ids=list(anchor.anchor_ids)) for task in base.tasks]
            base = CompilationResult(
                method="skyrescue_grounded_llm_candidate_v1_1",
                tasks=tasks,
                workflow_nodes=base.workflow_nodes,
                schema_valid=True,
                executable=True,
                failure=None,
                hallucinated_entity=False,
                unregistered_skill_call=False,
                permission_violation=False,
                latency_ms=base.latency_ms,
            )
        return GroundedV11Compilation(base, anchor)
    rejected = CompilationResult(
        method="skyrescue_grounded_llm_candidate_v1_1",
        tasks=[], workflow_nodes=[], schema_valid=True, executable=False,
        failure="UngroundedEntity", hallucinated_entity=True,
        unregistered_skill_call=False, permission_violation=False,
        latency_ms=base.latency_ms,
    )
    return GroundedV11Compilation(rejected, anchor)


__all__ = ["GrounderV11Anchor", "compile_grounded_candidate_v11", "ground_target_v11", "normalize_v11"]
