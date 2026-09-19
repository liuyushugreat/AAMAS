# SkyRescue (AAMAS Artifact)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reproducible implementation of **SkyRescue**: a checkable runtime contract for
typed admission of untrusted workflow candidates, crash-consistent commitment of
external effects, and commitment-preserving local repair on a causal impact
closure. The controlled evaluation domain is low-altitude emergency UAV mission
workflows; a synthetic DevOps adapter provides a second code-level portability
check. This is a single-process research prototype and does **not** claim
distributed exactly-once delivery, deployed UAV control, or real-flight
validation.

## Modules Directory

| Module | Path | Description |
|--------|------|-------------|
| **SkyRescue** | [`modules/Skyrescue/`](./modules/Skyrescue) | Typed IR admission, Proposal–Adjudication–Commit external-effect boundary, impact-closure repair, SkyRescue-Bench reproduction |

See [`modules/README.md`](./modules/README.md) for the full Modules Directory
index.

## Quick start

```bash
git clone https://github.com/liuyushugreat/AAMAS.git
cd AAMAS/modules/Skyrescue
python -m venv .venv-skyrescue
source .venv-skyrescue/bin/activate
pip install -r requirements-lock.txt
shasum -a 256 -c release/jss-submission-v1.0/SHA256SUMS.txt
PYTHONPATH=. python scripts/reproduce_jss_submission.py \
  --output-dir /tmp/skyrescue-jss-submission-v1.0
```

The offline pipeline verifies the frozen grounder, runs the full test suite,
re-scores stored HeldOut100 responses, executes 90 real process crashes, runs
the matched Native/LangGraph persistence experiment, the single-task-graph scale
experiment, and the two-domain contract check. The event oracle is
evaluator-only.

## What is covered

- **Typed admission**: symbol-anchored compiler boundary over untrusted language candidates
- **External-effect commitment**: Proposal–Adjudication–Commit with idempotency keys and execution receipts
- **Local repair**: commitment preservation outside the causal impact closure `IC(e,W)`
- **Framework embedding**: LangGraph wraps the shared repair/receiver contract; contract semantics remain explicit application logic
- **Scale and portability**: one connected typed workflow graph up to 5,000 tasks; synthetic UAV/DevOps adapters share one `RuntimeContract`

## License

This project is licensed under the Apache License 2.0. See [`LICENSE`](./LICENSE).
