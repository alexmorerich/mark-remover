# integration — the neutral merge coordinator

The integration area is **not one of the six agents**. It is the role that sits *above* them:
it assembles their outputs into one pipeline, runs the cross-agent end-to-end check, owns the
serial merge process, and **guards `shared/contract.py`** so the six agents can develop in parallel
without colliding on the one thing they all share.

> The six agents build in parallel. Integration is what makes their pieces fit together — and the
> gate that stops everyone editing the shared contract at once.

## What lives here

| File | Role |
|---|---|
| `pipeline.py` | Assembly: the ONLY module that imports all six concrete agents and wires them into an `Orchestrator`. Lives here (not in `agents/orchestrator/`) so no agent depends on another. |
| `e2e.py` | End-to-end smoke over real images — the check the integrator runs after merging branches, beyond the per-agent contract tests. |
| `README.md` | This charter + the merge / contract-change protocols. |

## Why it is separate from `agents/orchestrator/`

`agents/orchestrator/` is a **runtime** agent — one of the six — that routes/retries a single image.
Integration is a **build/merge-time** coordinator. Keeping assembly here means:

- No agent directory imports another agent (the orchestrator agent no longer depends on the five
  cleaners/detector/validator just to be wired up).
- A coder agent assigned to `orchestrator` cannot accidentally become the integrator of everyone
  else's work — the roles stay distinct.

## Merge protocol (owns agents/README.md §3.4)

1. Merge **low-risk single-agent fixes first** (small diff, no `shared/contract.py` change).
2. Run `tests/contract_tests/` **and** `integration/e2e.py` (end-to-end on a sample).
3. Merge **interface-touching changes last**, one at a time, re-running both test layers between each.

## Contract-change protocol (owns agents/README.md §4)

`shared/contract.py` is the single cross-agent coupling point. To stop six agents editing it at once:

- It changes **only** on a dedicated `contract/<change>` branch — never inside a feature/bug branch.
- That branch edits `shared/contract.py` + `tests/contract_tests/` together, is **reviewed by the
  integration owner**, merges first, then every agent rebases onto it.
- This is enforced in GitHub via [`.github/CODEOWNERS`](../.github/CODEOWNERS): `shared/` requires the
  integration owner's review. Turn on branch protection → *Require review from Code Owners* to make it
  binding.

## Run

```bash
python3 cli.py IMG.jpg --device mps                       # single image (uses integration.build_default_pipeline)
python3 -m integration.e2e IMAGES_DIR --n 20 --device cpu # post-merge end-to-end tally
python3 tests/contract_tests/test_contract.py             # per-agent boundary tests
```
