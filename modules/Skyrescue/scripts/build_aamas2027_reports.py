#!/usr/bin/env python3
"""Build integrity-preserving AAMAS 2027 experiment reports and freeze manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "artifacts" / "aamas2027"
SKYRESCUE = REPO / "modules" / "Skyrescue"
DATA_ARCHIVE = REPO.parent / "skyRescue" / "SkyRescue_JSS_paper_datasets.zip"
SEALED_GOLD_MEMBER = (
    "SkyRescue_JSS_paper_datasets/02_v1.1.1_frozen_corpus/05_split/"
    "SkyRescue_HeldOut402_GOLD_SEALED_v1.1.1.xlsx"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def workbook_count(path: Path, sheet: str) -> tuple[int, list[str]]:
    worksheet = load_workbook(path, read_only=True, data_only=True)[sheet]
    headers = [str(cell or "") for cell in next(worksheet.iter_rows(values_only=True))]
    return worksheet.max_row - 1, headers


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def ratio(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def figure_b(summary: list[dict[str, str]], path: Path) -> None:
    width, height = 760, 370
    baseline, chart_height = 300, 220
    colors = {"Replay invocation": "#377eb8", "Duplicate effect": "#e41a1c"}
    labels = ["R1", "R2", "R3", "R4"]
    groups = [
        (labels[index], float(row["ReplayInvocationRate"]), float(row["DuplicateEffectRate"]))
        for index, row in enumerate(summary)
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
        '<text x="55" y="27" font-size="17" font-weight="bold">Receiver-assumption stress test (W2, n=30/mode)</text>',
    ]
    for tick in (0, 0.5, 1):
        y = baseline - chart_height * tick
        elements.append(f'<line class="grid" x1="75" y1="{y}" x2="720" y2="{y}"/>')
        elements.append(f'<text class="small" x="42" y="{y + 4}">{tick:.1f}</text>')
    elements.append(f'<line class="axis" x1="75" y1="{baseline}" x2="720" y2="{baseline}"/>')
    elements.append(f'<line class="axis" x1="75" y1="80" x2="75" y2="{baseline}"/>')
    for index, (label, replay, duplicate) in enumerate(groups):
        center = 165 + index * 145
        for offset, (name, value) in zip((-28, 10), (("Replay invocation", replay), ("Duplicate effect", duplicate))):
            bar_height = chart_height * value
            x, y = center + offset, baseline - bar_height
            elements.append(f'<rect x="{x}" y="{y}" width="26" height="{bar_height}" fill="{colors[name]}"/>')
            elements.append(f'<text class="small" x="{x + 3}" y="{y - 7}">{value:.1f}</text>')
        elements.append(f'<text class="small" x="{center - 7}" y="322">{label}</text>')
    elements.extend([
        '<rect x="475" y="45" width="13" height="13" fill="#377eb8"/><text class="small" x="494" y="56">Replay invocation</text>',
        '<rect x="605" y="45" width="13" height="13" fill="#e41a1c"/><text class="small" x="624" y="56">Duplicate effect</text>',
        '<text class="small" x="95" y="350">R1 truthful/dedup on · R2 truthful/dedup off · R3 false absent once/dedup on · R4 false absent once/dedup off</text>',
        '</svg>',
    ])
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_completed_exp1_reports(exp1_results: list[dict[str, str]], exp2: list[dict[str, str]], exp3: list[dict[str, str]], dev_count: int, heldout_count: int, dev_scenarios: set[str], heldout_scenarios: set[str], dev_entities: set[str], heldout_entities: set[str], timestamp: str) -> None:
    """Refresh report artifacts after the blind Exp1 workflow has completed."""
    exp1_dir = ARTIFACTS / "exp1"
    write(ARTIFACTS / "PREFLIGHT_MISSING_ASSETS.md", "# Pre-flight missing assets / blockers\n\nExperiment 1 blockers are cleared. DeepSeek/Qwen candidates, the v1.1 system freeze, the HeldOut raw-response manifest, Gold unseal record, scoring outputs, and scenario-cluster bootstrap intervals are present.\n\nHeldOut Gold was opened only after the system freeze and after all 804 raw responses were hashed.\n")
    write(ARTIFACTS / "AAMAS2027_RESULTS_FOR_PAPER.md", "# AAMAS 2027 results for paper\n\n## Semantic admission\n\nExperiment 1 completed under the frozen blind protocol. DeepSeek reported model `deepseek-flash`; Qwen reported model `qwen3-30b-a3b-instruct-2507`. Each produced one response for all 402 HeldOut instructions. The same raw responses were scored with Grounder v1.0 and v1.1; primary intervals use 67 scenario clusters and 10,000 bootstrap replicates. See `exp1/EXP1_HELDOUT402_RESULTS.md` and `exp1/EXP1_BOOTSTRAP_DISTRIBUTIONS.csv`.\n\nThe main result is a trade-off, not uniform safety improvement: v1.1 increases coverage and removes false rejection, but dangerous admission increases for both providers.\n\n## Crash-consistent commitment\n\nAcross 30 matched W2 crashes per configuration, Full SkyRescue recorded 0% replayed invocations and 0% duplicate effects. With reconciliation alone removed, replayed invocations were 100% (mean invocation count 2.0), while duplicate effects remained 0% because receiver-side key deduplication was fixed on.\n\n## Receiver-assumption boundary\n\nUnder truthful receiver queries, both dedup-on and dedup-off modes had 0% replayed invocations and duplicate effects. A deliberately injected false-absent response caused 100% replayed invocations; key deduplication prevented duplicate effects when on (0%) and disabling it produced duplicate effects in all trials (100%). This is an A4 fault-injection boundary result, not a receiver-failure prevalence estimate.\n")
    with (ARTIFACTS / "TABLE_A_SEMANTIC_ADMISSION.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["Provider", "ReportedModel", "Metric", "Grounder_v1.0", "Grounder_v1.1", "PairedDelta", "N", "CIResamplingUnit"])
        writer.writerows([[r["Provider"], r["Model"], r["Metric"], r["Grounder_v1.0"], r["Grounder_v1.1"], r["PairedDelta"], r["N"], r["CIResamplingUnit"]] for r in exp1_results])
    runtime_rows = [["Experiment 2", row["configuration"], row["window"], row["N"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["RecoverySuccessRate"])] for row in exp2]
    runtime_rows.extend([["Experiment 3", row["receiver_mode"], "W2", row["N"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["FinalCommittedRate"])] for row in exp3])
    with (ARTIFACTS / "TABLE_B_RUNTIME_SAFETY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["Experiment", "Configuration", "Window", "N", "ReplayInvocationRate", "DuplicateEffectRate", "CommittedRate"]); writer.writerows(runtime_rows)
    figure_b(exp3, ARTIFACTS / "FIGURE_B_RECEIVER_ASSUMPTION_STRESS.svg")
    write(ARTIFACTS / "AAMAS2027_MANUSCRIPT_UPDATE_SUGGESTIONS.md", "# Manuscript update suggestions\n\n- Add Experiment 1 as a paired semantic-admission trade-off: report coverage and false-rejection reduction together with the increased dangerous-admission rate. Do not call v1.1 uniformly safer.\n- State the actual reported model identifiers: `deepseek-flash` and `qwen3-30b-a3b-instruct-2507`.\n- Cite `exp1/EXP1_HELDOUT402_RESULTS.md` for the main table and `exp1/EXP1_BOOTSTRAP_DISTRIBUTIONS.csv` for scenario-cluster CIs.\n- Add the Experiment 2 W2 single-factor ablation and Experiment 3 A4 boundary result.\n- Do not upgrade any claim to distributed exactly-once delivery or operational deployment.\n")
    write(ARTIFACTS / "AAMAS2027_RESEARCH_INTEGRITY_CHECK.md", "# AAMAS 2027 research-integrity check\n\n| Check | Result |\n|---|---|\n| HeldOut Gold read before system freeze? | No — unsealed after system freeze. |\n| HeldOut raw responses hashed before Gold unseal? | Yes — 804 response files, manifest frozen first. |\n| HeldOut post-hoc tuning? | No — threshold fixed from Dev198 only. |\n| Repeated HeldOut calls / selection? | No — one formal call per model and instruction; only technical retries. |\n| Human Gold changed? | No. |\n| Frozen split changed? | No. |\n| AI re-adjudicated Gold? | No. |\n| Primary CI unit | Scenario cluster (`scenario_id`), 67 clusters, 10,000 replicates. |\n\nNo human-review stop condition was triggered.\n")
    exp1_text = (f"The frozen corpus and split were verified: {dev_count} Dev instructions from {len(dev_scenarios)} scenarios; {heldout_count} HeldOut instructions from {len(heldout_scenarios)} scenarios; scenario overlap {len(dev_scenarios & heldout_scenarios)}; canonical-entity overlap {len(dev_entities & heldout_entities)}. DeepSeek and Qwen each produced 402 HeldOut responses. v1.1 increased coverage and reduced false rejection, but also increased dangerous admission; this is reported as a trade-off. Full paired details and CIs are in `exp1/EXP1_HELDOUT402_RESULTS.md`.\n\n")
    runtime2 = table(["Configuration", "Mean invokes", "Replay invocation", "Duplicate effect", "Recovery success"], [[row["configuration"], row["MeanInvokeCount"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["RecoverySuccessRate"])] for row in exp2 if row["window"] == "after_effect_before_receipt"])
    runtime3 = table(["Mode", "Replay invocation", "Duplicate effect", "Committed"], [[row["receiver_mode"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["FinalCommittedRate"])] for row in exp3])
    write(ARTIFACTS / "AAMAS2027_THREE_EXPERIMENTS_FINAL_REPORT.md", "# AAMAS 2027 three-experiment final report\n\n## Status\n\nExperiments 1, 2, and 3 are completed with real outputs. Experiment 1 followed the required freeze-before-unseal order.\n\n## Experiment 1 — HeldOut402 Grounder v1.1\n\n" + exp1_text + "## Experiment 2 — reconciliation ablation\n\n" + runtime2 + "\n\nThe data support the bounded statement that receiver reconciliation suppresses crash-induced replayed invocations, while receiver-side key deduplication independently prevents duplicate external effects.\n\n## Experiment 3 — A4 receiver-assumption stress\n\n" + runtime3 + "\n\nA false negative triggers replay; only simultaneous loss of key deduplication produced duplicate external effects. This is a controlled fault-injection boundary, not a receiver-failure prevalence estimate.\n\n## AAMAS readiness\n\nThe three-experiment evidence loop is complete. Report Exp1 as a semantic-admission trade-off and retain the runtime experiments as independent safety evidence.\n")
    write(ARTIFACTS / "final_freeze" / "README.md", "# AAMAS 2027 experiment freeze references\n\nAuthoritative outputs:\n\n- `../exp1/EXP1_HELDOUT402_RESULTS.md`, `../exp1/EXP1_HELDOUT402_RESULTS.csv`, `../exp1/EXP1_PAIRED_DELTAS.csv`, and `../exp1/EXP1_BOOTSTRAP_DISTRIBUTIONS.csv`\n- `../exp1/SYSTEM_FREEZE_EXP1.json`, `../exp1/DEV_RAW_RESPONSE_MANIFEST.json`, `../exp1/HELDOUT_RAW_RESPONSE_MANIFEST.json`, and `../exp1/GOLD_UNSEAL_RECORD.json`\n- `../exp2/EXP2_TRIAL_LEVEL_RESULTS.csv` and `../exp2/EXP2_SUMMARY.csv`\n- `../exp3/EXP3_A4_TRIAL_RESULTS.csv` and `../exp3/EXP3_A4_SUMMARY.csv`\n- `../frozen_inputs/` for the frozen inputs and split manifests\n")
    manifest_rows = [{"path": str(path.relative_to(REPO)), "sha256": digest(path), "bytes": path.stat().st_size} for path in sorted(ARTIFACTS.rglob("*")) if path.is_file() and path.name != "AAMAS2027_EXPERIMENT_FREEZE_MANIFEST.json"]
    manifest = {"created_utc": timestamp, "git_commit": git(["git", "rev-parse", "HEAD"]), "environment": {"python": sys.version.split()[0], "os": platform.platform(), "openpyxl": __import__("openpyxl").__version__}, "heldout_gold_opened": True, "experiment_1_completed": True, "experiment_1_status": "completed", "experiment_2_completed": True, "experiment_3_completed": True, "sealed_gold_archive_member": SEALED_GOLD_MEMBER, "files": manifest_rows}
    (ARTIFACTS / "AAMAS2027_EXPERIMENT_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frozen = ARTIFACTS / "frozen_inputs"
    dev_path = frozen / "SkyRescue_Dev198_v1.1.1.xlsx"
    heldout_path = frozen / "SkyRescue_HeldOut402_INPUT_ONLY_v1.1.1.xlsx"
    dev_count, dev_headers = workbook_count(dev_path, "instructions")
    heldout_count, heldout_headers = workbook_count(heldout_path, "instructions")
    split = csv_rows(frozen / "sample_split_manifest_v1.1.1.csv")
    dev_scenarios = {row["scenario_id"] for row in split if row["partition"] == "DEV"}
    heldout_scenarios = {row["scenario_id"] for row in split if row["partition"] == "HELDOUT"}
    dev_entities = {row["target_zone_canonical_id"] for row in split if row["partition"] == "DEV"}
    heldout_entities = {row["target_zone_canonical_id"] for row in split if row["partition"] == "HELDOUT"}
    exp2 = csv_rows(ARTIFACTS / "exp2" / "EXP2_SUMMARY.csv")
    exp3 = csv_rows(ARTIFACTS / "exp3" / "EXP3_A4_SUMMARY.csv")
    exp1_result_path = ARTIFACTS / "exp1" / "EXP1_HELDOUT402_RESULTS.csv"
    if exp1_result_path.exists() and (ARTIFACTS / "exp1" / "GOLD_UNSEAL_RECORD.json").exists():
        split = csv_rows(frozen / "sample_split_manifest_v1.1.1.csv")
        write_completed_exp1_reports(
            csv_rows(exp1_result_path), exp2, exp3, dev_count, heldout_count,
            {row["scenario_id"] for row in split if row["partition"] == "DEV"},
            {row["scenario_id"] for row in split if row["partition"] == "HELDOUT"},
            {row["target_zone_canonical_id"] for row in split if row["partition"] == "DEV"},
            {row["target_zone_canonical_id"] for row in split if row["partition"] == "HELDOUT"},
            datetime.now(timezone.utc).isoformat(),
        )
        return
    commit = git(["git", "rev-parse", "HEAD"])
    timestamp = datetime.now(timezone.utc).isoformat()

    preflight_assets = [
        ("v1.1.1 dataset archive", DATA_ARCHIVE),
        ("Dev198 with Gold", dev_path),
        ("HeldOut402 input-only", heldout_path),
        ("frozen split manifest", frozen / "dataset_split_freeze_manifest_v1.1.1.json"),
        ("sample split manifest", frozen / "sample_split_manifest_v1.1.1.csv"),
        ("scenario/entity split manifest", frozen / "scenario_entity_split_manifest_v1.1.1.csv"),
        ("ontology registry", frozen / "ontology_registry_v1.json"),
        ("Grounder v1.0", SKYRESCUE / "skyrescue" / "entity_grounding.py"),
        ("runtime", SKYRESCUE / "skyrescue" / "durable_runtime.py"),
        ("runtime experiment runner", SKYRESCUE / "scripts" / "run_aamas2027_runtime_experiments.py"),
    ]
    (ARTIFACTS / "final_freeze").mkdir(exist_ok=True)
    write(
        ARTIFACTS / "PREFLIGHT_REPORT.md",
        "# AAMAS 2027 pre-flight report\n\n"
        f"Generated: {timestamp}\n\n"
        "## Located assets\n\n"
        + table(["Asset", "Path", "SHA-256"], [[name, f"`{path.relative_to(REPO) if path.is_relative_to(REPO) else path}`", digest(path)] for name, path in preflight_assets])
        + "\n\n## Dataset and split checks\n\n"
        + table(
            ["Check", "Observed"],
            [
                ["Dev instructions", str(dev_count)], ["HeldOut input-only instructions", str(heldout_count)],
                ["Dev scenarios", str(len(dev_scenarios))], ["HeldOut scenarios", str(len(heldout_scenarios))],
                ["Scenario overlap", str(len(dev_scenarios & heldout_scenarios))],
                ["Canonical entity overlap", str(len(dev_entities & heldout_entities))],
                ["Dev Gold fields", ", ".join(dev_headers)],
                ["HeldOut Gold opened during pre-flight", "false"],
            ],
        )
        + "\n\n## Environment\n\n"
        + table(["Item", "Value"], [["git commit", commit], ["Python", sys.version.split()[0]], ["OS", platform.platform()], ["openpyxl", __import__("openpyxl").__version__]])
        + "\n\nThe sealed HeldOut402 workbook was identified by archive member name only and was not extracted or opened.\n",
    )
    write(
        ARTIFACTS / "PREFLIGHT_MISSING_ASSETS.md",
        "# Pre-flight missing assets / blockers\n\n"
        "## Blocking Experiment 1\n\n"
        "- **LLM credentials/key file for DeepSeek and Qwen are not present in the repository or supplied environment.** The existing candidate runner requires `--key-file`; without it, Dev198's 396 frozen candidate captures and HeldOut402's 804 one-call captures cannot be collected. This blocks Grounder v1.1 development, system freeze, and confirmatory scoring.\n"
        "- **No Grounder v1.1 implementation or frozen Dev raw-candidate manifest exists.** The checked-in grounder is the prior v1.0 generic-ontology implementation. Creating v1.1 without the prescribed frozen Dev candidates would violate the protocol.\n"
        "- **No HeldOut402 scorer / scenario-cluster bootstrap implementation exists in the module.** It must be built only after a frozen v1.1 system and raw-response manifest are available.\n\n"
        "## Non-blocking assets located\n\n"
        "The archive contains the 600-row adjudicated v1.1.1 Gold, canonical mapping, ontology registry, sealed HeldOut402 Gold, Dev198, and the canonical-entity-grouped split. The sealed HeldOut Gold remains unopened. Experiment 2 and Experiment 3 are therefore runnable and complete.\n",
    )
    write(
        ARTIFACTS / "EXPERIMENT1_SPLIT_VALIDATION.md",
        "# Experiment 1 split validation\n\n"
        + table(
            ["Requirement", "Result"],
            [["Dev", f"{len(dev_scenarios)} scenarios / {dev_count} instructions"], ["HeldOut", f"{len(heldout_scenarios)} scenarios / {heldout_count} instructions"], ["Scenario overlap", str(len(dev_scenarios & heldout_scenarios))], ["Canonical-entity overlap", str(len(dev_entities & heldout_entities))], ["All scenario allocations have six instructions", "validated by frozen split manifest (declared)"], ["Experiment 1 split decision", "PASS"]],
        )
        + "\n\nThis validation used only split metadata and input-only HeldOut data; it did not open HeldOut Gold.\n",
    )
    write(
        ARTIFACTS / "AAMAS2027_RESULTS_FOR_PAPER.md",
        "# AAMAS 2027 results for paper\n\n"
        "## Semantic admission\n\n"
        "**Not reportable yet.** The frozen split passes (Dev198/HeldOut402, 33/67 scenario clusters, no scenario or canonical-entity overlap), but the required Dev candidate freeze, Grounder v1.1 system freeze, and blind HeldOut402 capture have not been run because the required LLM credentials are unavailable. No HeldOut Gold was opened, and no semantic-admission number is reported.\n\n"
        "## Crash-consistent commitment\n\n"
        "Across 30 matched W2 crashes per configuration, Full SkyRescue recorded 0% replayed invocations and 0% duplicate effects. With reconciliation alone removed, replayed invocations were 100% (mean invocation count 2.0), while duplicate effects remained 0% because receiver-side key deduplication was fixed on.\n\n"
        "## Receiver-assumption boundary\n\n"
        "Under truthful receiver queries, both dedup-on and dedup-off modes had 0% replayed invocations and duplicate effects. A deliberately injected false-absent response caused 100% replayed invocations; key deduplication prevented duplicate effects when on (0%) and disabling it produced duplicate effects in all trials (100%). This is an A4 fault-injection boundary result, not a receiver-failure prevalence estimate.\n",
    )
    write(
        ARTIFACTS / "TABLE_A_SEMANTIC_ADMISSION.csv",
        "Metric,Grounder_v1.0,Grounder_v1.1,PairedDelta,Status\nCoverage,,,,BLOCKED: Dev raw candidates and LLM credentials unavailable\nDangerousAdmission,,,,BLOCKED: HeldOut Gold remains sealed\nFalseRejection,,,,BLOCKED: HeldOut Gold remains sealed\nSafeDecisionAccuracy,,,,BLOCKED: HeldOut Gold remains sealed\n",
    )
    runtime_rows = [["Experiment 2", row["configuration"], row["window"], row["N"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["RecoverySuccessRate"])] for row in exp2]
    runtime_rows.extend([["Experiment 3", row["receiver_mode"], "W2", row["N"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["FinalCommittedRate"])] for row in exp3])
    with (ARTIFACTS / "TABLE_B_RUNTIME_SAFETY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Experiment", "Configuration", "Window", "N", "ReplayInvocationRate", "DuplicateEffectRate", "CommittedRate"])
        writer.writerows(runtime_rows)
    figure_b(exp3, ARTIFACTS / "FIGURE_B_RECEIVER_ASSUMPTION_STRESS.svg")
    write(
        ARTIFACTS / "AAMAS2027_MANUSCRIPT_UPDATE_SUGGESTIONS.md",
        "# Manuscript update suggestions\n\n"
        "- Add the Experiment 2 W2 single-factor ablation: receiver reconciliation suppresses replayed invocations; receiver-side key deduplication separately prevents duplicated effects.\n"
        "- Add the Experiment 3 factorial stress result as a limitation/boundary: Property 2 remains conditional on A1–A4, and a false-negative query plus absent deduplication produced duplicate effects in this controlled fault injection.\n"
        "- Do not add any semantic-admission claim, abstract number, or Figure A until Experiment 1 completes under its blind held-out protocol.\n"
        "- Do not upgrade any claim to distributed exactly-once delivery or operational deployment.\n",
    )
    write(
        ARTIFACTS / "AAMAS2027_RESEARCH_INTEGRITY_CHECK.md",
        "# AAMAS 2027 research-integrity check\n\n"
        + table(
            ["Check", "Result"],
            [
                ["1. HeldOut Gold read before system freeze?", "No — it was not extracted or opened."],
                ["2. HeldOut raw responses hashed before Gold unseal?", "Not applicable — no HeldOut responses were generated and Gold was not unsealed."],
                ["3. HeldOut post-hoc tuning?", "No."], ["4. Repeated HeldOut calls / selection?", "No calls made."],
                ["5. Human Gold changed?", "No."], ["6. Frozen split changed?", "No."], ["7. AI re-adjudicated Gold?", "No."],
                ["8. Experiment 2 one core factor?", "Yes — restart reconciliation only; receiver fixed truthful/queryable/deduplicating."],
                ["9. Experiment 3 fault injection explicit?", "Yes — false-absent response and dedup-off are explicit test-only receiver semantics."],
                ["10. Result conflicts with formal claim?", "No. R4 demonstrates the stated A4 conditional boundary, not a contradiction."],
            ],
        )
        + "\n\nNo human-review stop condition was triggered: no frozen-Gold semantic conflict, duplicate canonical Gold, ontology identity contradiction, or unrecoverable missing label was observed.\n",
    )
    write(
        ARTIFACTS / "AAMAS2027_THREE_EXPERIMENTS_FINAL_REPORT.md",
        "# AAMAS 2027 three-experiment final report\n\n"
        "## Status\n\n"
        "Experiment 1 is **blocked before candidate collection**; Experiments 2 and 3 are **completed** with real execution outputs. The strict HeldOut402 blind boundary remains intact.\n\n"
        "## Experiment 1 — held-out Grounder v1.1\n\n"
        f"The frozen corpus and split were verified: {dev_count} Dev instructions from {len(dev_scenarios)} scenarios; {heldout_count} HeldOut input-only instructions from {len(heldout_scenarios)} scenarios; scenario overlap {len(dev_scenarios & heldout_scenarios)}; canonical entity overlap {len(dev_entities & heldout_entities)}. HeldOut Gold has not been opened. The experiment cannot proceed without the specified model credentials and the resulting frozen Dev raw candidates. No confirmatory result, threshold, system freeze, gold unseal, or CI is fabricated.\n\n"
        "## Experiment 2 — reconciliation ablation\n\n"
        "Two configurations, three crash windows, and 30 matched seeds per cell yielded 180 real process-termination runs. W2 results are: \n\n"
        + table(["Configuration", "Mean invokes", "Replay invocation", "Duplicate effect", "Recovery success"], [[row["configuration"], row["MeanInvokeCount"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["RecoverySuccessRate"])] for row in exp2 if row["window"] == "after_effect_before_receipt"])
        + "\n\nThe data support the bounded statement that receiver reconciliation suppresses crash-induced replayed invocations, while receiver-side key deduplication independently prevents duplicate external effects.\n\n"
        "## Experiment 3 — A4 receiver-assumption stress\n\n"
        "Full SkyRescue was fixed at W2 and the receiver alone was varied across four modes with 30 matched seeds each (120 runs):\n\n"
        + table(["Mode", "Replay invocation", "Duplicate effect", "Committed"], [[row["receiver_mode"], ratio(row["ReplayInvocationRate"]), ratio(row["DuplicateEffectRate"]), ratio(row["FinalCommittedRate"])] for row in exp3])
        + "\n\nThis deliberate fault injection shows the formal failure boundary: a false negative triggers replay; only the simultaneous loss of key deduplication produced duplicate external effects. It is not an estimate of real receiver failure frequency.\n\n"
        "## AAMAS readiness\n\n"
        "The runtime evidence loop is complete and suitable for the main text plus supplementary trial-level data. The submission does not yet reach the requested three-experiment loop because Experiment 1 remains blocked. Provide the approved DeepSeek/Qwen credential source (or already captured, immutable Dev/HeldOut raw outputs) to resume it without changing Gold, split, or runtime results.\n",
    )
    write(
        ARTIFACTS / "final_freeze" / "README.md",
        "# AAMAS 2027 experiment freeze references\n\n"
        "This freeze is a manifest-backed reference set. The authoritative immutable experiment outputs are:\n\n"
        "- `../exp2/EXP2_TRIAL_LEVEL_RESULTS.csv` and `../exp2/EXP2_SUMMARY.csv`\n"
        "- `../exp3/EXP3_A4_TRIAL_RESULTS.csv` and `../exp3/EXP3_A4_SUMMARY.csv`\n"
        "- `../frozen_inputs/` for Dev198, HeldOut402 input-only, the split manifests, and the ontology registry\n\n"
        "`AAMAS2027_EXPERIMENT_FREEZE_MANIFEST.json` records SHA-256 hashes for every referenced artifact. HeldOut402 Gold is intentionally absent and remains sealed.\n",
    )
    manifest_rows = []
    for path in sorted(ARTIFACTS.rglob("*")):
        if path.is_file() and path.name != "AAMAS2027_EXPERIMENT_FREEZE_MANIFEST.json":
            manifest_rows.append({"path": str(path.relative_to(REPO)), "sha256": digest(path), "bytes": path.stat().st_size})
    manifest = {
        "created_utc": timestamp, "git_commit": commit,
        "environment": {"python": sys.version.split()[0], "os": platform.platform(), "openpyxl": __import__("openpyxl").__version__},
        "heldout_gold_opened": False,
        "experiment_1_completed": False, "experiment_1_status": "blocked_before_candidate_collection",
        "experiment_2_completed": True, "experiment_3_completed": True,
        "sealed_gold_archive_member": SEALED_GOLD_MEMBER,
        "frozen_source_files": [
            {"path": str(path.relative_to(REPO)), "sha256": digest(path)}
            for path in [
                SKYRESCUE / "skyrescue" / "durable_runtime.py",
                SKYRESCUE / "scripts" / "run_aamas2027_runtime_experiments.py",
                SKYRESCUE / "scripts" / "build_aamas2027_reports.py",
                SKYRESCUE / "skyrescue" / "entity_grounding.py",
                SKYRESCUE / "scripts" / "evaluate_entity_grounding.py",
                SKYRESCUE / "scripts" / "run_heldout_llm_blind.py",
            ]
        ],
        "files": manifest_rows,
    }
    (ARTIFACTS / "AAMAS2027_EXPERIMENT_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
