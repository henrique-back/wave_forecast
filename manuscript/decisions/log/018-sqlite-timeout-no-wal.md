---
status: kept
date: 2026-08-06
commits: [b156e02]
category: training
---

# 018 — Optuna SQLite storage: `connect_args={'timeout': 30}`, deliberately not WAL mode

## Context

The 12h/24h/48h lead-time studies all share one `optuna_study_{STUDY_VERSION}.db` file and
are commonly launched as separate concurrent processes. Plain `sqlite:///` gives each
writer only Python `sqlite3`'s default 5-second busy-timeout, which is not always enough
under contention — a collision surfaces as `optuna.exceptions.StorageInternalError`
("exceeding max length" is just the generic message text) during a trial's final
state/value commit, crashing the whole run and leaving that trial stuck as `RUNNING`
forever.

## Change

Switched to `optuna.storages.RDBStorage(url=..., engine_kwargs={"connect_args":
{"timeout": 30}})`, so each connection waits out a lock instead of erroring immediately —
sufficient since contention windows here are brief per-trial commits, not sustained
overlapping writes. WAL mode (the more thorough general fix for concurrent SQLite writers)
was deliberately **not** adopted: the VS Code "Optuna Dashboard" extension reads a `.db`
file through a single-file sqlite-wasm VFS in the browser sandbox that can't follow a WAL
database's paired `.db-wal`/`.db-shm` side files, and its own bundled code
(`dist/web/storage.worker.js`) force-runs `pragma journal_mode=DELETE` the instant it opens
a file — pointing it at a WAL-mode db made it read a stale/incomplete single-file snapshot,
breaking the extension for the entire study history (not just new trials) the first time
this was tried.

## Evidence

First-hand account (no `results/` metrics applicable — this is an infrastructure/tooling
decision, not a model-performance one). No committed reproduction of the extension breakage
exists beyond the code comment's own description.

## Decision

Kept. `timeout=30` resolved the observed `StorageInternalError` crashes without the
tooling regression WAL mode caused when tried.

## Related

Code: `scripts/optimize.py` (storage construction)
Manuscript section this might feed: none (infrastructure detail, not a methodological
choice — out of scope per the parent CLAUDE.md's "write the science, not the repository"
rule).
