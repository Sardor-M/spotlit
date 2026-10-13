# web

The browser runtime: `src/tinylab.js`, one ES module with no dependencies. It reads a
`.tlab` file (`python/tinylab/export/README.md`) and embeds text on the calling thread.

```js
import { Student, cosine } from "./src/tinylab.js";

const student = Student.fromBuffer(await (await fetch("student.tlab")).arrayBuffer());
const score = cosine(student.embed("reciprocal rank fusion"), student.embed(passage));
```

Vectors are L2-normalized, so cosine is a dot product. Every vector shares a large first
component, so raw cosines sit high (unrelated texts score about 0.6); use them to rank, not
as calibrated similarities.

## Checks

```sh
uv run tinylab-export                 # writes dist/tinylab-student-v1
node web/test/parity.mjs              # encoder goldens, embedding goldens, timings
python3 -m http.server 8765 --bind 127.0.0.1   # then open /web/bench.html
```

`parity.mjs` checks the encoder against `tests/encoder_goldens.txt` and the embeddings
against the exporter's `goldens.json`. `bench.html` repeats both in a browser and times
embedding on the main thread.
