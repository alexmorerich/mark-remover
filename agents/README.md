# Multi-Agent Parallel Development — Watermark Remover

> **Goal:** six coder agents fix bugs / build features in parallel, with files that never collide
> and minimal integration risk.
> **Read before starting:** §0, §2, §4.

---

## 0. Core principle

Six directories give you **documentation / responsibility isolation** — not, on its own, **code
isolation**. If several agents all edit the same core file (`owner_agent.py` / `audit.py` /
`orchestrator.py`), they still collide. True parallelism needs **both**:

1. **Directory isolation** — each agent has its own working directory under `agents/`.
2. **Interface isolation** — each agent talks to the rest of the system ONLY through the typed
   interfaces in `shared/contract.py`; it never edits another agent's implementation.

> Directory isolation decides *whether you can write in parallel*; interface isolation decides
> *whether the merge blows up*.

---

## 1. Layout

```
agents/
  detector/            # detection logic only
  orchestrator/        # routing / retry / state machine + pipeline assembly only
  classic_cleaner/     # OpenCV / Telea cleaner only          (charter dir: "classic-cleaner")
  neural_cleaner/      # LaMa cleaner only                    (charter dir: "neural-cleaner")
  diffusion_cleaner/   # tier-3 stub / interface only         (charter dir: "diffusion-cleaner")
  validator/           # QA checks only
shared/
  contract.py          # shared typed record / enum / interface — changes need coordination (§4)
tests/
  contract_tests/      # integration boundary; every agent must keep it green
```

> **Naming note.** Python package directories use underscores (`classic_cleaner`) because hyphens
> are not valid import identifiers; the agent **charters** and the canonical agent **names**
> (`classic-cleaner`) use hyphens. Same agent, two spellings — code vs. prose.

Each agent directory carries its own `AGENT.md` charter (mission · inputs · outputs · boundaries ·
definition of done).

## Pipeline

```text
Images
  └─▶ detector ─(no watermark)─▶ skip
        │ mask + type + score
        ▼
     orchestrator ─(after N failed retries)─▶ manual_review
        │ route by type · risk · score · retry history
        ├──────────────┬───────────────┐
        ▼              ▼               ▼
  classic_cleaner  neural_cleaner  diffusion_cleaner
     tier 1           tier 2           tier 3
     OpenCV            LaMa          SD / Imagen (stub)
        └──────────────┴───────────────┘
                       ▼
                   validator ──pass──▶ pass
                       └────fail────▶ orchestrator retry
```

---

## 2. Responsibilities and red lines

| Agent | Owns | Output contract | Never does |
|---|---|---|---|
| **detector** | detection | `mask` / `watermark_type` / `score` | no pixel edits, no routing |
| **orchestrator** | route / retry / state machine | scheduling decision | no pixels, no cleaner internals |
| **classic_cleaner** | OpenCV cleaner | cleaned result (input unchanged) | no detection, no QA |
| **neural_cleaner** | LaMa cleaner | cleaned result (input unchanged) | same |
| **diffusion_cleaner** | tier-3 stub / interface | cleaned result | same |
| **validator** | QA checks | `pass` / `fail` only | no pixel edits, no detection |

**General rules**

- An agent may change **only its own directory** plus the tests it needs.
- **Do not** edit `shared/contract.py` directly; interface changes are coordinated separately (§4).
- Any change must keep `tests/contract_tests/` **all green** before it is committed.

---

## 3. Parallel bug-fix workflow

### 3.1 One branch / worktree per agent

```bash
git worktree add .worktrees/detector-bug       -b codex/detector-bug
git worktree add .worktrees/validator-bug      -b codex/validator-bug
git worktree add .worktrees/neural-cleaner-bug -b codex/neural-cleaner-bug
# … one isolated branch per agent
```

> Worktrees share one `.git` object store — cheap, no re-clone. ⚠️ The **same branch cannot be
> checked out in two worktrees**; every agent uses its own branch.

### 3.2 Each agent works only in its own worktree

Detector → mask bug · Validator → QA misjudgement · Cleaner → patch scar. Each touches only its own
directory — no file is stepped on twice.

### 3.3 Contract tests are the integration boundary

Before any commit, all of these must hold (and are asserted in `tests/contract_tests/`):

- **detector** outputs `mask` / `watermark_type` / `score`.
- **cleaners** never modify the original; they only return a cleaned result.
- **validator** returns only `pass` / `fail` (a `QAReport`); it never repairs pixels.
- **orchestrator** only routes / retries; it never touches pixels.

> If the contract tests pass, each agent's output composes with the others.

### 3.4 Integration merges serially

1. Merge **low-risk single-agent fixes** first (small diff, no interface change).
2. Run **end-to-end** tests.
3. Merge interface-touching changes last.

---

## 4. Changing `shared/contract.py`

`shared/contract.py` is the one shared coupling point — touching it affects every agent.

- ❌ Don't slip a contract change into a normal bug branch.
- ✅ Open a dedicated `contract/<change>` branch: edit `contract.py` + `contract_tests` there, merge
  it first, then every agent rebases onto it.
- ✅ Contract changes are reviewed — never a silent single-agent edit.

---

## 5. Landing strategy: additive typed layer

This layer **wraps** the stable engines rather than rewriting them:

- Each agent directory is a thin **wrapper / adapter** over the corresponding proven implementation
  (`detector` → `logo_finder`; `classic_cleaner` → `product_preserve_clean` / cv2 Telea;
  `neural_cleaner` → `run_bulk` LaMa; `validator` → `audit.py`).
- It exposes those through the typed interface in `shared/contract.py`.
- The old files (`owner_agent.py` / `audit.py` / top-level `orchestrator.py`) stay stable as the
  fallback baseline.

So bugs fan out to agents in parallel, conflicts stay confined to the thin adapter layer, and the
legacy code remains a controllable rollback point.

### Two reconciliations with the frozen production contract

- **`manual_review` ≡ `auto_rejected`.** CONTRACT_v1 guarantees no human in the per-image loop: a
  failed image is `auto_rejected` and its original restored from backup — that **is** the single
  failure exit. `Status.MANUAL_REVIEW.terminal_label` returns `auto_rejected` for manifest
  compatibility. No review queue is added.
- **`diffusion_cleaner` is a registered stub** → routes to `manual_review`. It satisfies the
  `Cleaner` interface so the ladder + open/closed registry are real and tested; drop a real backend
  into its `clean()` to activate tier 3 with no orchestrator change.

## Shared handoff record

Agents communicate through one per-image record (typed Python models in-process; may also be JSONL):

```json
{
  "image_id": "image001",
  "detector":     { "has_watermark": true, "mask_path": "...", "bbox": [120,80,460,130],
                    "watermark_type": "semi_transparent_text", "score": 0.93 },
  "orchestrator": { "route": "neural-cleaner", "retry_count": 0, "max_retries": 3 },
  "cleaner":      { "agent": "neural-cleaner", "status": "cleaned" },
  "validator":    { "verdict": "pass", "qa_score": 0.98, "failed_checks": [] },
  "terminal": "pass"
}
```

## Terminal states

| State | Meaning |
|---|---|
| `skip` | detector found no watermark; image untouched |
| `pass` | validator approved the cleaned image (`published`) |
| `retry` | validator failed and retry budget remains (internal, not terminal) |
| `manual_review` | retry budget exhausted / too risky (`auto_rejected`, original restored) |

## Run

```bash
python3 cli.py IMG.jpg --device mps [--heuristic] [--out clean.jpg]   # single image
python3 tests/contract_tests/test_contract.py                         # contract tests (pure, no GPU)
```
