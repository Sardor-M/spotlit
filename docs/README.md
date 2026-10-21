# Documentation

Public documentation lives here.

- [benchmarks/](benchmarks/README.md): every measurement behind the release
- [reproduce.md](reproduce.md): every command from the Mind2Web download to the released files
- [strengths-and-limits.md](strengths-and-limits.md): what spotlit is good at and where it falls short, with evidence
- [../models/spotlit-v1/](../models/spotlit-v1/README.md): the model card
- [../python/tinylab/export/README.md](../python/tinylab/export/README.md): the `.tlab` file format
- [../web/README.md](../web/README.md): the browser runtime and API

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
