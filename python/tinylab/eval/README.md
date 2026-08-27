# tinylab.eval

The evaluation harness. Built before the model, deliberately — the Moka teardown's
step 3. Zero dependencies, stdlib only.

Nothing here knows what an embedding is. A retriever is anything with `index(docs)` and
`rank(query_text, top_k)`, so the student model will be scored by this same code path
against numbers recorded before it existed.

## Data format

Two files per evaluation set:

```
corpus.jsonl    {"doc_id": str, "unit_id": str, "text": str}
queries.jsonl   {"query_id": str, "text": str, "relevant": {doc_id: grade}}
```

`unit_id` is the correlated unit — the page or document a passage was cut from.
Passages from one source are near-duplicates, so splits are assigned per **unit**,
never per passage. `grade` is a non-negative int; 1 is relevant, higher is more
relevant. nDCG uses grades, recall ignores them.

A query's judged docs must all fall in one split, or loading fails. Otherwise the query
would be training data and held-out data at once.

## Splits

Assigned by a stable hash of `unit_id`, so adding documents later cannot reshuffle what
is already assigned.

| Bucket | Split | Use |
| :--- | :--- | :--- |
| 0-9 | val | freely |
| 10-19 | test | rarely |
| 20-29 | sealed | once, ever — see `docs/all/experiments/SEALED.md` |
| 30-99 | train | — |

Retrieval always runs over the **whole** corpus, because that is what deployment looks
like. Only the queries are filtered by split. No label is used at retrieval time.

## Commands

```
tinylab-eval splits                                  # split distribution
tinylab-eval run --system bm25 --split val           # score and record
tinylab-eval compare --incumbent ID --candidate ID   # keep/discard gate
tinylab-eval verify-harness                          # frozen files intact?
tinylab-eval selfcheck                               # harness sanity, no ledger write
```

Point at a real set with `--set path/to/dir`. Without it, the bundled fixture is used.

`run` refuses a configuration it has already recorded. Pass `--force` only when you mean
it.

## The frozen files

`schema.py`, `splits.py`, `metrics.py`, `harness.py`, `rigor.py` are hashed in
`FROZEN.sha256`. Every ledger entry records whether they were intact when the number was
produced. Changing them needs `bless-harness` plus an ADR — see ADR-0002.

## Fixture vs the real set

`fixtures/` is a 31-doc smoke test. It proves the plumbing and the baseline ordering and
nothing else — it saturates, so its absolute numbers are meaningless.

The real set is built from Lumen's docs and is not committed (see ADR-0004):

```sh
python3 -m tinylab.data.build_lumen_set \
    --source ../lumen/docs --out data/lumen
tinylab-eval --set data/lumen run --system bm25 --split val
```

Val split, 27 queries over 389 passages:

| System | nDCG@10 | recall@10 | MRR |
| :--- | ---: | ---: | ---: |
| random | 0.0000 | 0.0000 | 0.0038 |
| lexical-overlap | 0.1976 | 0.2741 | 0.1832 |
| bm25 | 0.2435 | 0.3407 | 0.2220 |
| dense (teacher) | 0.3373 | 0.5333 | 0.3033 |
| hybrid (RRF) | 0.3483 | 0.4963 | 0.3244 |

`teacher` and `hybrid` need the optional extra and are imported lazily, so the stdlib-only
path never pulls in torch:

```sh
uv sync --extra teacher
PYTHONPATH=python .venv/bin/python -m tinylab.eval.cli \
    --set data/lumen run --system hybrid --split val
```

See ADR-0003 for what these numbers do and do not establish.

## Next

The student. It registers as another retriever and is scored by this same code path,
against numbers recorded before it existed.
