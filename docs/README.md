# Documentation

Public documentation lives here. Design notes that ship with the project — format specs,
model cards, embedding guides — go in this directory as they are written.

## Private notes

`docs/all/` is gitignored and stays local. It holds working material:

```
docs/all/
├── learnings/      teardowns of prior art, and lessons organized by stack
├── decisions/      ADRs — what was chosen and why
└── experiments/    the running experiment log
```

That separation is intentional. Working notes are written to build understanding and are
revised constantly; published documentation is written for someone else and should be
stable. Mixing them means either the notes get sanitized into uselessness or the public
docs get cluttered with dead ends.

If a note in `docs/all/` matures into something worth publishing, move it here.
