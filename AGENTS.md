# AGENTS.md

Rules for anyone — human or agent — writing code in this repository.

## Comments: short, or absent

**Never write long explanatory comment blocks in source files.** No multi-paragraph
docstrings, no teaching essays above a function, no restating the design in prose.
Code carries the logic; comments carry only what the code cannot say.

The limits, and they are hard limits:

| Where | Maximum | Only when |
| :--- | :--- | :--- |
| Module docstring | 1 line, 2 if genuinely needed | The filename isn't self-explanatory |
| Function / method docstring | 1 line | The signature doesn't already say it |
| Inline comment | 1 line | A non-obvious *why* — a constant's origin, a subtle bug avoided |
| Class docstring | 1 line | — |

Never do this:

```python
"""Make the goalposts structurally hard to move.

Moka forbade the optimizing agent from editing `rigor.py`, `arena.py`, `board.py`
and `teacher.py`. A rule in a prompt is a suggestion; a hash manifest is a fact.

`verify()` reports which frozen files no longer match `FROZEN.sha256`. Every recorded
run carries the answer, so a number produced while the judge was modified is
identifiable forever after.
"""
```

Do this:

```python
"""Hash manifest for the frozen evaluation files."""
```

Also banned:

- `# ---- section banner ----` dividers
- Comments that repeat the line below them (`# increment counter`)
- Commented-out code — delete it, git remembers
- TODO essays; one line with a pointer, or an issue instead
- Comments explaining *what changed* — that is the commit message's job

## Where the reasoning goes instead

Rationale is valuable. It just doesn't live in source files.

| Kind of writing | Home |
| :--- | :--- |
| Why an approach was chosen, alternatives rejected | `docs/all/decisions/` (ADR) |
| How a subsystem works, how to use it | that directory's `README.md` |
| What a measurement showed | `docs/all/experiments/ledger.jsonl` |
| Reference-project teardowns and learning notes | `docs/all/learnings/` |

If you catch yourself writing a third comment line, stop: that paragraph belongs in an
ADR or a README, and the code should get a one-line pointer to it at most.

## Why this rule exists

A file that is half prose is a file nobody reads. Comments rot faster than code, and a
stale explanation is worse than no explanation — it actively misleads. Prose in
`docs/` can be revised as one document; prose scattered across forty files cannot.

## Naming and structure

- Names carry the meaning a comment would have. Rename before you annotate.
- A function that needs paragraphs to explain is too big; split it.
- Prefer one clear expression over a clever one plus a comment defending it.

## Repository conventions

- `docs/all/` is gitignored — private notes, learnings, decisions, experiment logs.
- The evaluation harness has **zero dependencies** and stays that way.
- Files listed in `python/tinylab/eval/FROZEN.sha256` are the judge. Don't edit them to
  make a number look better; if they must change, re-bless and write an ADR.
- No emoji or icon decoration in docs or code. Plain text.
