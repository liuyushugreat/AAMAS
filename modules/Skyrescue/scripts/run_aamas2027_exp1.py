#!/usr/bin/env python3
"""Capture and score the v1.1 Dev198/HeldOut402 confirmatory experiment.

This script deliberately uses a dataset-aligned schema (the frozen v1.1.1
corpus does not use the older human-intent benchmark vocabulary).  The
``capture`` command makes exactly one formal call per model and instruction,
with only technical transport retries.  The ``freeze-dev`` command selects a
pre-registered threshold using Dev Gold.  The ``score-heldout`` command must
be run only after the raw HeldOut manifest and system freeze exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import time
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from run_human_intent_llm_benchmark import PROVIDERS, load_secret, post_json
from skyrescue.entity_grounding import compile_grounded_candidate, ground_target
from skyrescue.grounder_v1_1 import compile_grounded_candidate_v11, normalize_v11
from skyrescue.workflow import compile_generated_candidate


FIELDS = ("task_type", "target_zone", "priority", "deadline_s_or_text", "required_skill", "needs_human_approval", "expected_failure")
ENUMS = {
    "task_type": {"modify_existing_task", "multi_area_search", "person_search", "emergency_supply_delivery", "area_patrol", "medical_delivery"},
    "priority": {"HIGH", "NORMAL"},
    "deadline_s_or_text": {"未明确", "尽快", "600"},
    "required_skill": {"search", "delivery", "workflow_repair", "surveillance"},
    "needs_human_approval": {"True", "False"},
    "expected_failure": {"None", "TemporalConflict", "HumanApprovalRequired", "UngroundedEntity"},
}
SYSTEM_PROMPT = """你是 SkyRescue 低空应急指令结构化抽取器。依据场景背景和指挥指令，输出一个 JSON 对象，不得输出 Markdown、解释或额外字段。必须恰好包含：task_type, target_zone, priority, deadline_s_or_text, required_skill, needs_human_approval, expected_failure。
允许值：task_type=modify_existing_task|multi_area_search|person_search|emergency_supply_delivery|area_patrol|medical_delivery；priority=HIGH|NORMAL；deadline_s_or_text=未明确|尽快|600；required_skill=search|delivery|workflow_repair|surveillance；needs_human_approval=True|False；expected_failure=None|TemporalConflict|HumanApprovalRequired|UngroundedEntity。
target_zone 必须填写指令中最具体的地点短语；无法唯一确认地点时使用指令语义对应的地点文本并将 expected_failure 设为 UngroundedEntity。只返回 JSON。"""
USER_TEMPLATE = "场景背景：{scenario_card}\n指挥指令：{instruction_text}"
PROMPT_VERSION = "SkyRescue-AAMAS2027-Exp1-v1.1.0"
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_ITERATIONS = 10000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def workbook_rows(path: Path, sheet: str) -> list[dict[str, Any]]:
    worksheet = load_workbook(path, read_only=True, data_only=True)[sheet]
    rows = list(worksheet.values)
    headers = [str(value or "") for value in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:] if any(value is not None for value in row)]


def build_inputs(dev_path: Path, heldout_path: Path, output_dir: Path) -> None:
    dev_instructions = workbook_rows(dev_path, "instructions")
    dev_gold = workbook_rows(dev_path, "gold")
    heldout_instructions = workbook_rows(heldout_path, "instructions")
    def cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{
            "instruction_id": str(row["样本ID"]),
            "scenario_id": str(row["来源Scenario"]),
            "scenario_card": str(row["Scenario Card情境"]),
            "instruction_text": str(row["自然语言指令"]),
        } for row in rows]
    dev_cases = cases(dev_instructions)
    heldout_cases = cases(heldout_instructions)
    by_id = {row["样本ID"]: row for row in dev_instructions}
    gold_rows = []
    for row in dev_gold:
        source = by_id[str(row["样本ID"])]
        gold_rows.append({
            "instruction_id": str(row["样本ID"]),
            "scenario_id": str(row["场景实例ID"]),
            "scenario_card": str(source["Scenario Card情境"]),
            "instruction_text": str(row["自然语言指令"]),
            "task_type": str(row["task_type"]),
            "target_zone": str(row["target_zone"]),
            "target_zone_surface": str(row["target_zone_surface"]),
            "target_zone_canonical_id": str(row["target_zone_canonical_id"]),
            "target_zone_canonical_name": str(row["target_zone_canonical_name"]),
            "priority": str(row["priority"]),
            "deadline_s_or_text": str(row["deadline_s_or_text"]),
            "required_skill": str(row["required_skill"]),
            "needs_human_approval": str(row["needs_human_approval"]),
            "expected_failure": str(row["expected_failure"]),
        })
    write_jsonl(output_dir / "dev_cases.jsonl", dev_cases)
    write_jsonl(output_dir / "heldout_cases.jsonl", heldout_cases)
    write_jsonl(output_dir / "dev_gold.jsonl", gold_rows)
    (output_dir / "input_manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dev_cases": len(dev_cases), "heldout_cases": len(heldout_cases), "dev_gold": len(gold_rows),
        "dev_input_sha256": sha256_path(dev_path), "heldout_input_sha256": sha256_path(heldout_path),
        "gold_labels_sent_to_model": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_heldout_gold(gold_path: Path, input_path: Path, output_path: Path) -> None:
    gold_rows = workbook_rows(gold_path, "gold")
    input_rows = {row["instruction_id"]: row for row in read_jsonl(input_path)}
    converted = []
    for row in gold_rows:
        sample_id = str(row["样本ID"])
        source = input_rows[sample_id]
        converted.append({
            "instruction_id": sample_id, "scenario_id": str(row["场景实例ID"]),
            "scenario_card": source["scenario_card"], "instruction_text": source["instruction_text"],
            "task_type": str(row["task_type"]), "target_zone": str(row["target_zone"]),
            "target_zone_surface": str(row["target_zone_surface"]), "target_zone_canonical_id": str(row["target_zone_canonical_id"]),
            "target_zone_canonical_name": str(row["target_zone_canonical_name"]), "priority": str(row["priority"]),
            "deadline_s_or_text": str(row["deadline_s_or_text"]), "required_skill": str(row["required_skill"]),
            "needs_human_approval": str(row["needs_human_approval"]), "expected_failure": str(row["expected_failure"]),
        })
    write_jsonl(output_path, converted)
    print(json.dumps({"gold_rows": len(converted), "gold_sha256": sha256_path(gold_path)}, ensure_ascii=False, indent=2))


def schema_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["root_not_object"]
    errors: list[str] = []
    expected = set(FIELDS)
    if set(value) != expected:
        errors.extend(["missing:" + ",".join(sorted(expected - set(value)))]) if expected - set(value) else None
        errors.extend(["extra:" + ",".join(sorted(set(value) - expected))]) if set(value) - expected else None
    for field, allowed in ENUMS.items():
        if field in value and (not isinstance(value[field], str) or value[field] not in allowed):
            errors.append(f"enum:{field}")
    if "target_zone" in value and not str(value["target_zone"]).strip():
        errors.append("empty:target_zone")
    return errors


def strict_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg}@{exc.pos}"
    if not isinstance(parsed, dict):
        return None, "root_not_object"
    # The frozen Gold stores booleans/nulls and numeric deadlines, while the
    # public extraction schema uses canonical strings.  This deterministic
    # parser coercion is part of the frozen post-processing step; raw content
    # remains preserved verbatim in each capture record.
    normalized = dict(parsed)
    if isinstance(normalized.get("needs_human_approval"), bool):
        normalized["needs_human_approval"] = "True" if normalized["needs_human_approval"] else "False"
    if normalized.get("expected_failure") is None:
        normalized["expected_failure"] = "None"
    if isinstance(normalized.get("deadline_s_or_text"), (int, float)) and not isinstance(normalized.get("deadline_s_or_text"), bool):
        normalized["deadline_s_or_text"] = str(int(normalized["deadline_s_or_text"]))
    return normalized, None


def call_one(provider: str, case: dict[str, Any], key: str) -> dict[str, Any]:
    config = PROVIDERS[provider]
    prompt = USER_TEMPLATE.format(**case)
    payload: dict[str, Any] = {
        "model": config["model"], "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "max_tokens": 512, "stream": False,
    }
    if provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    started = time.perf_counter()
    response = None
    error = None
    attempts = 0
    for attempts in range(1, 5):
        try:
            response = post_json(config["url"], key, payload)
            break
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempts == 4:
                break
        except Exception as exc:  # network-dependent
            error = f"{type(exc).__name__}: {exc}"
            if attempts == 4:
                break
        time.sleep(2 ** (attempts - 1))
    base = {
        "experiment": PROMPT_VERSION, "provider": provider, "requested_model": config["model"],
        "instruction_id": case["instruction_id"], "scenario_id": case["scenario_id"],
        "scenario_card_sent": True, "gold_labels_sent": False, "attempts": attempts,
        "prompt_sha256": sha256_bytes((SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")),
        "instruction_sha256": sha256_bytes(case["instruction_text"].encode("utf-8")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if response is None:
        return {**base, "api_success": False, "error": error}
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        return {**base, "api_success": False, "error": f"malformed_response:{exc}"}
    parsed, parse_error = strict_json(content)
    errors = schema_errors(parsed) if parsed is not None else ["json_parse_failed"]
    return {**base, "api_success": True, "reported_model": response.get("model"), "usage": response.get("usage", {}),
            "content": content, "json_parse_success": parsed is not None, "json_parse_error": parse_error,
            "parsed": parsed, "schema_valid": not errors, "schema_errors": errors}


def capture(input_path: Path, key_file: Path, output_dir: Path, workers: int) -> None:
    cases = read_jsonl(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps({
        "experiment": PROMPT_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.resolve()), "input_sha256": sha256_path(input_path), "cases": len(cases),
        "prompt_version": PROMPT_VERSION, "prompt_sha256": sha256_bytes((SYSTEM_PROMPT + "\n" + USER_TEMPLATE).encode("utf-8")),
        "parameters": {"temperature": 0, "top_p": 1, "max_tokens": 512, "runs_per_case": 1, "technical_retry_max": 4},
        "secrets_persisted": False, "gold_labels_sent": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for provider in ("deepseek", "qwen"):
        key = load_secret(PROVIDERS[provider]["env"], key_file)
        path = output_dir / f"raw_{provider}.jsonl"
        existing = {row["instruction_id"] for row in read_jsonl(path) if row.get("api_success")} if path.exists() else set()
        pending = [case for case in cases if case["instruction_id"] not in existing]
        print(f"{provider}: {len(existing)} cached, {len(pending)} pending", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(call_one, provider, case, key) for case in pending]
            for index, future in enumerate(as_completed(futures), 1):
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                if index % 10 == 0 or index == len(pending):
                    print(f"{provider}: completed {index}/{len(pending)}", flush=True)
        rows = read_jsonl(path)
        latest = {row["instruction_id"]: row for row in rows}
        path.write_text("".join(json.dumps(latest[key], ensure_ascii=False) + "\n" for key in sorted(latest)), encoding="utf-8")
    summary = []
    for provider in ("deepseek", "qwen"):
        rows = read_jsonl(output_dir / f"raw_{provider}.jsonl")
        summary.append({"provider": provider, "cases": len(rows), "api_success": sum(bool(row.get("api_success")) for row in rows),
                        "json_parse": sum(bool(row.get("json_parse_success")) for row in rows), "schema_valid": sum(bool(row.get("schema_valid")) for row in rows),
                        "reported_models": sorted({row.get("reported_model") for row in rows if row.get("reported_model")})})
    (output_dir / "capture_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_capture(capture_dir: Path) -> dict[str, list[dict[str, Any]]]:
    captures = {}
    for provider in ("deepseek", "qwen"):
        rows = read_jsonl(capture_dir / f"raw_{provider}.jsonl")
        for row in rows:
            if row.get("api_success") and isinstance(row.get("content"), str):
                parsed, parse_error = strict_json(row["content"])
                errors = schema_errors(parsed) if parsed is not None else ["json_parse_failed"]
                row["parsed"] = parsed
                row["json_parse_success"] = parsed is not None
                row["json_parse_error"] = parse_error
                row["schema_valid"] = not errors
                row["schema_errors"] = errors
        captures[provider] = rows
    return captures


def normalize_aliases(gold: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in gold:
        cid = row["target_zone_canonical_id"]
        aliases[cid].update({row["target_zone"], row["target_zone_surface"], row["target_zone_canonical_name"]})
    return {key: tuple(sorted(values)) for key, values in sorted(aliases.items())}


def admission_metrics(rows: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]], threshold: float, aliases: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    count = len(rows)
    accepted_v10 = accepted_v11 = 0
    correct_v10 = correct_v11 = 0
    safe_v10 = safe_v11 = 0
    correct_total = 0
    wrong_accept_v10 = wrong_accept_v11 = 0
    correct_reject_v10 = correct_reject_v11 = 0
    for response in rows:
        gold = gold_by_id[response["instruction_id"]]
        parsed = response.get("parsed") if response.get("schema_valid") else None
        pred = parsed or {}
        predicted_target = str(pred.get("target_zone", ""))
        gold_target = normalize_v11(gold["target_zone"])
        is_correct = bool(parsed) and normalize_v11(predicted_target) in {gold_target, normalize_v11(gold["target_zone_surface"]), normalize_v11(gold["target_zone_canonical_name"])}
        correct_total += int(is_correct)
        v10_anchor = ground_target(predicted_target, gold["scenario_card"], gold["instruction_text"])
        v11_anchor = __import__("skyrescue.grounder_v1_1", fromlist=["ground_target_v11"]).ground_target_v11(predicted_target, gold["scenario_card"], gold["instruction_text"], threshold=threshold, aliases=aliases)
        v10_accept = bool(parsed) and v10_anchor.resolved
        v11_accept = bool(parsed) and v11_anchor.resolved
        accepted_v10 += int(v10_accept); accepted_v11 += int(v11_accept)
        correct_v10 += int(v10_accept and is_correct); correct_v11 += int(v11_accept and is_correct)
        wrong_accept_v10 += int(v10_accept and not is_correct); wrong_accept_v11 += int(v11_accept and not is_correct)
        correct_reject_v10 += int(is_correct and not v10_accept); correct_reject_v11 += int(is_correct and not v11_accept)
        gold_exec = gold["expected_failure"] == "None"
        # The v1.1 corpus has a distinct task vocabulary from the older
        # workflow compiler.  Its frozen execution oracle is the structured
        # decision pair in the candidate itself: no declared failure and no
        # human-approval gate means executable; otherwise it is blocked.
        direct_exec = bool(parsed and pred.get("expected_failure") == "None" and pred.get("needs_human_approval") == "False")
        exec_v10 = bool(direct_exec and v10_accept)
        exec_v11 = bool(direct_exec and v11_accept)
        safe_v10 += int(exec_v10 == gold_exec); safe_v11 += int(exec_v11 == gold_exec)
    incorrect_total = count - correct_total
    return {
        "instructions": count,
        "coverage": accepted_v11 / count if count else 0.0,
        "dangerous_admission": wrong_accept_v11 / count if count else 0.0,
        "conditional_dangerous_admission": wrong_accept_v11 / incorrect_total if incorrect_total else 0.0,
        "false_rejection": correct_reject_v11 / correct_total if correct_total else 0.0,
        "safe_decision_accuracy": safe_v11 / count if count else 0.0,
        "v10_coverage": accepted_v10 / count if count else 0.0,
        "v10_dangerous_admission": wrong_accept_v10 / count if count else 0.0,
        "v10_false_rejection": correct_reject_v10 / correct_total if correct_total else 0.0,
        "v10_safe_decision_accuracy": safe_v10 / count if count else 0.0,
        "canonical_accuracy": correct_total / count if count else 0.0,
        "accepted_correct_v11": correct_v11, "accepted_incorrect_v11": wrong_accept_v11,
        "correct_total": correct_total, "incorrect_total": incorrect_total,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def freeze_dev(capture_dir: Path, inputs_dir: Path, output_dir: Path) -> None:
    gold = read_jsonl(inputs_dir / "dev_gold.jsonl")
    gold_by_id = {row["instruction_id"]: row for row in gold}
    captures = load_capture(capture_dir)
    raw_root = output_dir / "dev_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    aliases = normalize_aliases(gold)
    thresholds = [0.60, 0.70, 0.80, 0.90]
    dev_rows = {provider: captures[provider] for provider in captures}
    threshold_rows = []
    for threshold in thresholds:
        metrics = []
        for provider in ("deepseek", "qwen"):
            metrics.append(admission_metrics(dev_rows[provider], gold_by_id, threshold, aliases))
        threshold_rows.append({"threshold": threshold, "dangerous_admission": max(row["dangerous_admission"] for row in metrics), "coverage": statistics.mean(row["coverage"] for row in metrics), "safe_decision_accuracy": statistics.mean(row["safe_decision_accuracy"] for row in metrics)})
    selected = sorted(threshold_rows, key=lambda row: (row["dangerous_admission"], -row["coverage"], row["threshold"]))[0]
    rule_text = "Select the threshold with the lowest Dev dangerous admission across providers; break ties by higher mean Dev coverage; final tie by lower threshold. Candidate set: 0.60, 0.70, 0.80, 0.90."
    (output_dir / "GROUNDING_THRESHOLD_SELECTION_RULE.md").write_text("# Exp1 threshold selection rule\n\n" + rule_text + f"\n\nSelected threshold: `{selected['threshold']:.2f}`. Selection used Dev198 only.\n", encoding="utf-8")
    aliases_path = output_dir / "grounder_v1_1_aliases.json"
    aliases_path.write_text(json.dumps({"source": "Dev198 Gold target_zone/target_zone_surface/target_zone_canonical_name", "aliases": aliases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "DEV_THRESHOLD_SELECTION.csv", threshold_rows)
    dev_metric_rows = []
    for provider in ("deepseek", "qwen"):
        metric = admission_metrics(dev_rows[provider], gold_by_id, float(selected["threshold"]), aliases)
        metric["provider"] = provider; metric["threshold"] = selected["threshold"]; dev_metric_rows.append(metric)
    write_csv(output_dir / "DEV_GROUNDER_SELECTION_METRICS.csv", dev_metric_rows)
    for provider, rows in captures.items():
        provider_dir = raw_root / provider; provider_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            (provider_dir / f"{row['instruction_id']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_rows = []
    for path in sorted(raw_root.rglob("*.json")):
        manifest_rows.append({"path": str(path.relative_to(output_dir)), "sha256": sha256_path(path), "bytes": path.stat().st_size})
    (output_dir / "DEV_RAW_RESPONSE_MANIFEST.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "cases": 198, "providers": ["deepseek", "qwen"], "raw_outputs_frozen": True, "files": manifest_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    freeze_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "system_frozen": True,
        "grounder_v1_0_version": sha256_path(Path(__file__).resolve().parents[1] / "skyrescue" / "entity_grounding.py"),
        "grounder_v1_1_version": sha256_path(Path(__file__).resolve().parents[1] / "skyrescue" / "grounder_v1_1.py"),
        "threshold": selected["threshold"], "threshold_selection_rule": rule_text,
        "threshold_selection_rule_hash": sha256_path(output_dir / "GROUNDING_THRESHOLD_SELECTION_RULE.md"),
        "aliases_hash": sha256_path(aliases_path), "dev_raw_manifest_hash": sha256_path(output_dir / "DEV_RAW_RESPONSE_MANIFEST.json"),
        "dev_input_hash": sha256_path(inputs_dir / "dev_cases.jsonl"), "dev_gold_hash": sha256_path(inputs_dir / "dev_gold.jsonl"),
        "prompt_version": PROMPT_VERSION, "prompt_hash": sha256_bytes((SYSTEM_PROMPT + "\n" + USER_TEMPLATE).encode("utf-8")),
        "parameters": {"temperature": 0, "top_p": 1, "max_tokens": 512, "runs_per_case": 1},
        "parser_version": sha256_path(Path(__file__).resolve()), "postprocessor_version": sha256_path(Path(__file__).resolve()),
        "ontology_version": "contextual-v1.1-dev-aliases", "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_gold_opened": False,
    }
    (output_dir / "SYSTEM_FREEZE_EXP1.json").write_text(json.dumps(freeze_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_threshold": selected["threshold"], "dev_metrics": dev_metric_rows}, ensure_ascii=False, indent=2))


def metric_from_rows(rows: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]], threshold: float, aliases: dict[str, tuple[str, ...]], version: str) -> dict[str, float]:
    accepted = wrong = correct_reject = correct = safe = 0
    for response in rows:
        gold = gold_by_id[response["instruction_id"]]
        parsed = response.get("parsed") if response.get("schema_valid") else None
        target = str((parsed or {}).get("target_zone", ""))
        is_correct = bool(parsed) and normalize_v11(target) in {normalize_v11(gold["target_zone"]), normalize_v11(gold["target_zone_surface"]), normalize_v11(gold["target_zone_canonical_name"])}
        correct += int(is_correct)
        anchor = (ground_target(target, gold["scenario_card"], gold["instruction_text"]) if version == "v1.0" else __import__("skyrescue.grounder_v1_1", fromlist=["ground_target_v11"]).ground_target_v11(target, gold["scenario_card"], gold["instruction_text"], threshold=threshold, aliases=aliases))
        accept = bool(parsed) and anchor.resolved
        accepted += int(accept); wrong += int(accept and not is_correct); correct_reject += int(is_correct and not accept)
        predicted_exec = bool(parsed and (parsed or {}).get("expected_failure") == "None" and (parsed or {}).get("needs_human_approval") == "False" and accept)
        safe += int(predicted_exec == (gold["expected_failure"] == "None"))
    n = len(rows); incorrect = n - correct
    return {"Coverage": accepted / n if n else 0.0, "DangerousAdmission": wrong / n if n else 0.0,
            "ConditionalDangerousAdmission": wrong / incorrect if incorrect else 0.0, "FalseRejection": correct_reject / correct if correct else 0.0,
            "SafeDecisionAccuracy": safe / n if n else 0.0, "CanonicalGroundingAccuracy": correct / n if n else 0.0}


def bootstrap(rows: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]], threshold: float, aliases: dict[str, tuple[str, ...]], version: str) -> dict[str, list[float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[gold_by_id[row["instruction_id"]]["scenario_id"]].append(row)
    scenario_ids = sorted(grouped)
    # Precompute each instruction's frozen decision once.  Bootstrap then only
    # resamples six-row scenario clusters; it must not re-run string matching
    # 10,000 times per provider/version.
    annotated = []
    for row in rows:
        gold = gold_by_id[row["instruction_id"]]
        parsed = row.get("parsed") if row.get("schema_valid") else None
        target = str((parsed or {}).get("target_zone", ""))
        correct = bool(parsed) and normalize_v11(target) in {normalize_v11(gold["target_zone"]), normalize_v11(gold["target_zone_surface"]), normalize_v11(gold["target_zone_canonical_name"])}
        anchor = ground_target(target, gold["scenario_card"], gold["instruction_text"]) if version == "v1.0" else __import__("skyrescue.grounder_v1_1", fromlist=["ground_target_v11"]).ground_target_v11(target, gold["scenario_card"], gold["instruction_text"], threshold=threshold, aliases=aliases)
        accepted = bool(parsed) and anchor.resolved
        predicted_exec = bool(parsed and parsed.get("expected_failure") == "None" and parsed.get("needs_human_approval") == "False" and accepted)
        annotated.append({"scenario": gold["scenario_id"], "accepted": accepted, "correct": correct, "safe": predicted_exec == (gold["expected_failure"] == "None")})
    by_scenario = defaultdict(list)
    for item in annotated:
        by_scenario[item["scenario"]].append(item)
    rng = random.Random(BOOTSTRAP_SEED)
    distributions = {metric: [] for metric in ("Coverage", "DangerousAdmission", "ConditionalDangerousAdmission", "FalseRejection", "SafeDecisionAccuracy", "CanonicalGroundingAccuracy")}
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = [item for scenario in (rng.choice(scenario_ids) for _ in scenario_ids) for item in by_scenario[scenario]]
        n = len(sampled); accepted = sum(item["accepted"] for item in sampled); correct = sum(item["correct"] for item in sampled); wrong = sum(item["accepted"] and not item["correct"] for item in sampled); safe = sum(item["safe"] for item in sampled); incorrect = n - correct
        metrics = {"Coverage": accepted / n if n else 0.0, "DangerousAdmission": wrong / n if n else 0.0, "ConditionalDangerousAdmission": wrong / incorrect if incorrect else 0.0, "FalseRejection": (correct - (accepted - wrong)) / correct if correct else 0.0, "SafeDecisionAccuracy": safe / n if n else 0.0, "CanonicalGroundingAccuracy": correct / n if n else 0.0}
        for metric, value in metrics.items(): distributions[metric].append(value)
    return distributions


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values); pos = (len(ordered) - 1) * fraction; lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def score_heldout(capture_dir: Path, inputs_dir: Path, gold_path: Path, output_dir: Path) -> None:
    freeze_path = output_dir / "SYSTEM_FREEZE_EXP1.json"
    if not freeze_path.exists(): raise RuntimeError("SYSTEM_FREEZE_EXP1.json is required before held-out scoring")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")); threshold = float(freeze["threshold"])
    gold = read_jsonl(gold_path); gold_by_id = {row["instruction_id"]: row for row in gold}; aliases_data = json.loads((output_dir / "grounder_v1_1_aliases.json").read_text(encoding="utf-8"))["aliases"]; aliases = {key: tuple(value) for key, value in aliases_data.items()}
    captures = load_capture(capture_dir)
    results = []; bootstrap_rows = []; detail_rows = []
    for provider in ("deepseek", "qwen"):
        rows = captures[provider]
        v10 = metric_from_rows(rows, gold_by_id, threshold, aliases, "v1.0")
        v11 = metric_from_rows(rows, gold_by_id, threshold, aliases, "v1.1")
        for metric in v10:
            results.append({"Provider": provider, "Model": rows[0].get("reported_model") or rows[0].get("requested_model"), "Metric": metric, "Grounder_v1.0": round(v10[metric], 6), "Grounder_v1.1": round(v11[metric], 6), "PairedDelta": round(v11[metric] - v10[metric], 6), "N": len(rows), "CIResamplingUnit": "scenario_id"})
        dists = {"v1.0": bootstrap(rows, gold_by_id, threshold, aliases, "v1.0"), "v1.1": bootstrap(rows, gold_by_id, threshold, aliases, "v1.1")}
        for version, by_metric in dists.items():
            for metric, values in by_metric.items():
                bootstrap_rows.append({"Provider": provider, "Version": version, "Metric": metric, "Estimate": round(metric_from_rows(rows, gold_by_id, threshold, aliases, version)[metric], 6), "CI95Low": round(percentile(values, .025), 6), "CI95High": round(percentile(values, .975), 6), "BootstrapIterations": BOOTSTRAP_ITERATIONS, "BootstrapSeed": BOOTSTRAP_SEED, "ResamplingUnit": "scenario_id"})
        for response in rows:
            gold_row = gold_by_id[response["instruction_id"]]; parsed = response.get("parsed") if response.get("schema_valid") else None; target = str((parsed or {}).get("target_zone", "")); correct = bool(parsed) and normalize_v11(target) in {normalize_v11(gold_row["target_zone"]), normalize_v11(gold_row["target_zone_surface"]), normalize_v11(gold_row["target_zone_canonical_name"])}
            detail_rows.append({"Provider": provider, "Model": response.get("reported_model") or response.get("requested_model"), "InstructionID": response["instruction_id"], "ScenarioID": gold_row["scenario_id"], "PredictedTarget": target, "GoldTarget": gold_row["target_zone"], "CanonicalCorrect": correct, "V10Accepted": bool(parsed and ground_target(target, gold_row["scenario_card"], gold_row["instruction_text"]).resolved), "V11Accepted": bool(parsed and __import__("skyrescue.grounder_v1_1", fromlist=["ground_target_v11"]).ground_target_v11(target, gold_row["scenario_card"], gold_row["instruction_text"], threshold=threshold, aliases=aliases).resolved)})
    write_csv(output_dir / "EXP1_HELDOUT402_RESULTS.csv", results); write_csv(output_dir / "EXP1_BOOTSTRAP_DISTRIBUTIONS.csv", bootstrap_rows); write_csv(output_dir / "EXP1_HELDOUT402_DETAILS.csv", detail_rows)
    by_metric = {(row["Provider"], row["Metric"]): row for row in results}
    paired = [{"Provider": row["Provider"], "Metric": row["Metric"], "Delta": row["PairedDelta"]} for row in results]
    write_csv(output_dir / "EXP1_PAIRED_DELTAS.csv", paired)
    md = ["# Experiment 1 — HeldOut402 Grounder v1.1 confirmatory evaluation", "", f"HeldOut Gold was opened only after `SYSTEM_FREEZE_EXP1.json` and frozen raw responses. Threshold={threshold:.2f}; n=402 instructions; primary CI unit=scenario_id (67 clusters), 10,000 percentile bootstrap replicates.", "", "| Provider | Metric | v1.0 | v1.1 | Paired Δ | 95% CI v1.1 |", "|---|---|---:|---:|---:|---:|"]
    ci_map = {(row["Provider"], row["Version"], row["Metric"]): row for row in bootstrap_rows}
    for row in results:
        ci = ci_map[(row["Provider"], "v1.1", row["Metric"])]
        md.append(f"| {row['Provider']} | {row['Metric']} | {row['Grounder_v1.0']:.2%} | {row['Grounder_v1.1']:.2%} | {row['PairedDelta']:+.2%} | [{ci['CI95Low']:.2%}, {ci['CI95High']:.2%}] |")
    md += ["", "The two grounders use the same frozen raw model responses. Canonical correctness is an offline comparison against the sealed Gold's target surface/canonical name; Gold labels were never supplied to the model or online grounder.", ""]
    (output_dir / "EXP1_HELDOUT402_RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "GOLD_UNSEAL_RECORD.json").write_text(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "system_freeze_hash": sha256_path(freeze_path), "heldout_gold_hash": sha256_path(gold_path), "raw_outputs_frozen": True, "heldout_gold_opened": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_heldout(capture_dir: Path, inputs_dir: Path, output_dir: Path) -> None:
    expected = {row["instruction_id"] for row in read_jsonl(inputs_dir / "heldout_cases.jsonl")}
    files = []
    for provider in ("deepseek", "qwen"):
        rows = read_jsonl(capture_dir / f"raw_{provider}.jsonl")
        ids = [row.get("instruction_id") for row in rows if row.get("api_success")]
        if set(ids) != expected or len(ids) != len(set(ids)):
            raise RuntimeError(f"{provider} raw capture incomplete or duplicated: {len(ids)} successful rows")
        provider_dir = output_dir / "heldout_raw" / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            path = provider_dir / f"{row['instruction_id']}.json"
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            files.append({"provider": provider, "instruction_id": row["instruction_id"], "path": str(path.relative_to(output_dir)), "sha256": sha256_path(path), "bytes": path.stat().st_size})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": len(expected), "providers": ["deepseek", "qwen"], "raw_outputs_frozen": True, "gold_opened_before_manifest": False, "input_sha256": sha256_path(inputs_dir / "heldout_cases.jsonl"), "files": files}
    (output_dir / "HELDOUT_RAW_RESPONSE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(expected), "files": len(files), "manifest_sha256": sha256_path(output_dir / "HELDOUT_RAW_RESPONSE_MANIFEST.json")}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--dev", type=Path, required=True); p.add_argument("--heldout", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("extract-gold"); p.add_argument("--gold-xlsx", type=Path, required=True); p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("capture"); p.add_argument("--input", type=Path, required=True); p.add_argument("--key-file", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("freeze-dev"); p.add_argument("--capture-dir", type=Path, required=True); p.add_argument("--inputs-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("score-heldout"); p.add_argument("--capture-dir", type=Path, required=True); p.add_argument("--inputs-dir", type=Path, required=True); p.add_argument("--gold", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("freeze-heldout"); p.add_argument("--capture-dir", type=Path, required=True); p.add_argument("--inputs-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": build_inputs(args.dev, args.heldout, args.output_dir)
    elif args.command == "extract-gold": extract_heldout_gold(args.gold_xlsx, args.input, args.output)
    elif args.command == "capture": capture(args.input, args.key_file, args.output_dir, args.workers)
    elif args.command == "freeze-dev": freeze_dev(args.capture_dir, args.inputs_dir, args.output_dir)
    elif args.command == "score-heldout": score_heldout(args.capture_dir, args.inputs_dir, args.gold, args.output_dir)
    elif args.command == "freeze-heldout": freeze_heldout(args.capture_dir, args.inputs_dir, args.output_dir)


if __name__ == "__main__":
    main()
