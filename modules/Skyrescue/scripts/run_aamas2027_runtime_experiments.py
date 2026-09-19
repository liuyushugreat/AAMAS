#!/usr/bin/env python3
"""Run AAMAS 2027 runtime ablation and A4 fault-injection experiments.

This runner uses real child-process termination in the existing SQLite runtime.
Experiment 2 changes only restart reconciliation; Experiment 3 fixes the full
runtime and changes only the simulated receiver's query correctness and
key-deduplication semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyrescue.durable_runtime import CRASH_EXIT_CODE, CrashPoint, DurableWorkflowRuntime


EXP2_CONFIGS = {
    "FullSkyRescue": {"reconciliation_mode": "full"},
    "RetryOnMissingReceipt": {"reconciliation_mode": "retry_on_missing_receipt"},
}
EXP3_CONFIGS = {
    "R1_TRUTHFUL_DEDUP_ON": {
        "receiver_query_mode": "truthful",
        "receiver_deduplicates": True,
    },
    "R2_TRUTHFUL_DEDUP_OFF": {
        "receiver_query_mode": "truthful",
        "receiver_deduplicates": False,
    },
    "R3_FALSE_ABSENT_ONCE_DEDUP_ON": {
        "receiver_query_mode": "false_absent_once",
        "receiver_deduplicates": True,
    },
    "R4_FALSE_ABSENT_ONCE_DEDUP_OFF": {
        "receiver_query_mode": "false_absent_once",
        "receiver_deduplicates": False,
    },
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))]


def runtime(database: Path, config: dict[str, Any]) -> DurableWorkflowRuntime:
    return DurableWorkflowRuntime(database, **config)


def worker(database: Path, workflow_id: str, crash_point: str | None, config: dict[str, Any]) -> None:
    instance = runtime(database, config)
    try:
        instance.start(workflow_id)
        instance.execute(workflow_id, crash_point=crash_point)
    finally:
        if crash_point is None:
            instance.close()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_trial(
    *,
    output_dir: Path,
    trial_id: int,
    seed: int,
    label: str,
    window: CrashPoint,
    config: dict[str, Any],
) -> dict[str, Any]:
    database = output_dir / "databases" / f"{label}_{window.value}_{trial_id:03d}.sqlite"
    workflow_id = f"{label}-{window.value}-{trial_id:03d}"
    config_json = json.dumps(config, sort_keys=True)
    worker_command = [
        sys.executable,
        __file__,
        "--worker",
        "--database",
        str(database),
        "--workflow-id",
        workflow_id,
        "--crash-point",
        window.value,
        "--config-json",
        config_json,
    ]
    crashed = subprocess.run(worker_command, check=False)
    if crashed.returncode != CRASH_EXIT_CODE:
        raise RuntimeError(f"{label} {window.value} trial {trial_id} crash returned {crashed.returncode}")
    started = time.perf_counter()
    resumed = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            "--database",
            str(database),
            "--workflow-id",
            workflow_id,
            "--config-json",
            config_json,
        ],
        check=False,
    )
    recovery_ms = (time.perf_counter() - started) * 1000
    instance = runtime(database, config)
    try:
        state = instance.inspect(workflow_id)
    finally:
        instance.close()
    if resumed.returncode != 0:
        raise RuntimeError(f"{label} {window.value} trial {trial_id} resume returned {resumed.returncode}")
    return {
        "trial_id": f"trial_{trial_id:03d}",
        "seed": seed,
        "configuration": label,
        "window": window.value,
        "invoke_count": state["invoke_count"],
        "effect_count": state["effect_count"],
        "receipt_count": state["receipt_count"],
        "duplicate_invocation": state["invoke_count"] > 1,
        "duplicate_effect": state["effect_count"] > 1,
        "receipt_violation": not state["local_receipt_valid"] or state["receipt_count"] != 1,
        "final_state": state["operation_state"],
        "workflow_version": state["workflow_version"],
        "reservation_consistent": state["reservation_consistent"],
        "receipt_binding_valid": state["receiver_receipt_valid"],
        "evidence_chain_valid": state["evidence_chain_continuous"],
        "recovery_latency_ms": round(recovery_ms, 4),
        "receiver_query_count": state["receiver_query_count"],
        "receiver_query_result": state["receiver_query_result"] or "not_queried",
        "retry_count": max(0, state["invoke_count"] - 1),
        "crash_exit_code": crashed.returncode,
        "resume_exit_code": resumed.returncode,
    }


def proportion(rows: list[dict[str, Any]], field: str, expected: Any = True) -> float:
    return round(sum(row[field] == expected for row in rows) / len(rows), 6)


def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    summaries = []
    for key in sorted(groups):
        group = groups[key]
        entry = dict(zip(group_fields, key))
        entry.update(
            {
                "N": len(group),
                "MeanInvokeCount": round(sum(row["invoke_count"] for row in group) / len(group), 6),
                "ReplayInvocationRate": proportion(group, "duplicate_invocation"),
                "DuplicateEffectRate": proportion(group, "duplicate_effect"),
                "ReceiptViolationRate": proportion(group, "receipt_violation"),
                "RecoverySuccessRate": proportion(group, "final_state", "Committed"),
                "P50RecoveryLatencyMs": round(percentile([row["recovery_latency_ms"] for row in group], 0.50), 4),
                "P95RecoveryLatencyMs": round(percentile([row["recovery_latency_ms"] for row in group], 0.95), 4),
            }
        )
        summaries.append(entry)
    return summaries


def run_exp2(output_dir: Path, trials: int) -> None:
    rows = []
    seeds = [202709000 + trial for trial in range(1, trials + 1)]
    for label, variant in EXP2_CONFIGS.items():
        config = {"receiver_deduplicates": True, "receiver_query_mode": "truthful", **variant}
        for window in CrashPoint:
            for trial_id, seed in enumerate(seeds, start=1):
                rows.append(
                    run_trial(
                        output_dir=output_dir,
                        trial_id=trial_id,
                        seed=seed,
                        label=label,
                        window=window,
                        config=config,
                    )
                )
    fields = list(rows[0])
    write_csv(output_dir / "EXP2_TRIAL_LEVEL_RESULTS.csv", rows, fields)
    summary = summarize(rows, ["configuration", "window"])
    write_csv(output_dir / "EXP2_SUMMARY.csv", summary, list(summary[0]))
    w2 = [row for row in summary if row["window"] == CrashPoint.AFTER_EFFECT_BEFORE_RECEIPT.value]
    (output_dir / "EXP2_COMMIT_ABLATION_REPORT.md").write_text(
        "# Experiment 2 — Single-factor commit/reconciliation ablation\n\n"
        f"All {len(rows)} runs use real child-process termination, matched trial seeds, the same SQLite persistence engine, receiver, idempotency key, workflow, reservations, and crash schedule. The only changed factor is restart reconciliation: `FullSkyRescue` queries the receiver; `RetryOnMissingReceipt` skips that query and reissues the same logical operation. The receiver remains truthful, queryable, and key-deduplicating in both configurations.\n\n"
        "A replayed invocation is not a duplicated external effect: the latter is separately measured as `effect_count > 1`.\n\n"
        "## W2 result\n\n"
        + "\n".join(
            f"- {row['configuration']}: mean invocations {row['MeanInvokeCount']:.3f}; replay-invocation rate {row['ReplayInvocationRate']:.3f}; duplicate-effect rate {row['DuplicateEffectRate']:.3f}."
            for row in w2
        )
        + "\n\nThe output supports the bounded mechanism claim only when these observed W2 rates differ as stated; it does not claim distributed exactly-once delivery.\n",
        encoding="utf-8",
    )


def run_exp3(output_dir: Path, trials: int) -> None:
    rows = []
    seeds = [202709000 + trial for trial in range(1, trials + 1)]
    for label, config in EXP3_CONFIGS.items():
        for trial_id, seed in enumerate(seeds, start=1):
            row = run_trial(
                output_dir=output_dir,
                trial_id=trial_id,
                seed=seed,
                label=label,
                window=CrashPoint.AFTER_EFFECT_BEFORE_RECEIPT,
                config={"reconciliation_mode": "full", **config},
            )
            row["receiver_mode"] = label
            row["query_correctness"] = "FALSE_ABSENT_ONCE" if config["receiver_query_mode"] != "truthful" else "TRUTHFUL"
            row["dedup"] = "ON" if config["receiver_deduplicates"] else "OFF"
            row["actual_precrash_effect_state"] = "occurred"
            row["query_returned_state"] = row["receiver_query_result"]
            row["human_escalated"] = row["final_state"] == "HumanEscalated"
            rows.append(row)
    fields = list(rows[0])
    write_csv(output_dir / "EXP3_A4_TRIAL_RESULTS.csv", rows, fields)
    summary = summarize(rows, ["receiver_mode", "query_correctness", "dedup"])
    for entry in summary:
        members = [row for row in rows if row["receiver_mode"] == entry["receiver_mode"]]
        entry["HumanEscalationRate"] = proportion(members, "human_escalated")
        entry["FinalCommittedRate"] = proportion(members, "final_state", "Committed")
    write_csv(output_dir / "EXP3_A4_SUMMARY.csv", summary, list(summary[0]))
    (output_dir / "EXP3_A4_STRESS_REPORT.md").write_text(
        "# Experiment 3 — A4 receiver-assumption stress test\n\n"
        f"This is an assumption-violation fault injection, not a normal-environment benchmark or an estimate of receiver-failure probability. It fixes Full SkyRescue and the W2 crash point, then varies only receiver query correctness and key deduplication across {len(rows)} matched-seed runs.\n\n"
        "A false-absent response is deliberately injected only into the first recovery reconciliation query after an effect has occurred.\n\n"
        "## Result\n\n"
        + "\n".join(
            f"- {row['receiver_mode']}: replay-invocation {row['ReplayInvocationRate']:.3f}; duplicate-effect {row['DuplicateEffectRate']:.3f}; committed {row['FinalCommittedRate']:.3f}."
            for row in summary
        )
        + "\n\nWhen A4 is deliberately violated, these results describe the observed formal failure boundary; they do not change the conditional scope of Property 2.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--experiment", choices=["exp2", "exp3", "all"], default="all")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--workflow-id")
    parser.add_argument("--crash-point", choices=[point.value for point in CrashPoint])
    parser.add_argument("--config-json")
    args = parser.parse_args()
    if args.worker:
        worker(args.database, args.workflow_id, args.crash_point, json.loads(args.config_json))
        return
    if args.trials <= 0:
        parser.error("--trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "databases").mkdir(exist_ok=True)
    if args.experiment in {"exp2", "all"}:
        run_exp2(args.output_dir, args.trials)
    if args.experiment in {"exp3", "all"}:
        run_exp3(args.output_dir, args.trials)
    shutil.rmtree(args.output_dir / "databases")


if __name__ == "__main__":
    main()
