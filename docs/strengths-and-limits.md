# Strengths, weaknesses and limitations

What spotlit v1 is good at and where it falls short, with the measurement behind each line.
Numbers are from [benchmarks/](benchmarks/README.md).

**Strengths**

| Strength | Evidence |
| :--- | :--- |
| Small enough to ship with a page or an extension | 6.7 MB in three files. INT8 export moved dev recall@50 by +0.6 points |
| No server and no GPU | WebAssembly in Chrome: 236 ms per page step on 8 threads, about 1 s on one |
| Cuts what the agent sends | The top 50 is 32-43x fewer tokens than the cleaned HTML on the test splits |
| Holds up on websites it never saw | Cross-website and cross-domain recall@50 of 75.8% and 75.9%, 31-32 points above BM25. Dev agrees with these splits within 3 points |
| Tolerant of spelling and formatting | Stage 1 reads bytes: "Tokio tee ceremony" and "Tokyo tea ceremony" have cosine 0.69, and "Sign in" and "sign-in" are identical after normalising |
| Same answers in every runtime | Python, JavaScript and C++ agree: 31/31 encoder goldens, 667/667 WordPiece pairs, top 50 on 40/40 dev steps in Chrome |
| Open and reproducible | Apache-2.0 weights, MIT code, public training data, and every variant that failed is documented |

**Weaknesses**

| Weakness | Evidence | What it means |
| :--- | :--- | :--- |
| A shortlist, not a picker | Dev recall@1 is 10.9% and recall@10 is 49.3% | The agent's model still has to choose among the 50 |
| Misses about one step in four on new websites | Cross-website recall@50 is 75.8%; recall@100 is 82.3% | An agent needs a fallback: a larger k, or the full page |
| Behind larger rankers | MindAct's 184M-parameter ranker reports 8-10 points more recall@50 | BERT-mini (13.4 MB) was 6 points better on dev but took 1.2 s per step |
| Stage 1 caps stage 2 | Stage 1's top 400 holds the clicked element on 87.6% of dev steps | The other 12.4% can't be recovered by the re-ranker |
| Stage 1 memorizes websites | Its recall@50 on unseen websites stopped near 54% after 11k updates while training loss kept falling | The gains on new websites come from stage 2's pretraining |
| Near-synonyms and look-alikes | For "Sign in to my account", stage 2 ranks "Sign up free" first and "Log in" fifth | Unusual wording on a site can mislead it |

**Limitations**

| Limitation | Detail |
| :--- | :--- |
| English websites from 2023 | Trained on Mind2Web's English sites. Other languages and newer page structures are untested |
| Text only | Each element is one line of text. No screenshot, position, size or visibility |
| 128 tokens per pair | Stage 2 truncates the goal and element to 128 WordPiece tokens |
| Candidates from one document | `collectCandidates` reads up to 2,000 elements with one selector query. Elements inside iframes and shadow roots are not collected |
| Ranking only | It does not plan, scroll, fill forms or decide when a task is done |
| WebAssembly only for INT8 | ONNX Runtime Web 1.30's WebGPU backend returns wrong scores for the INT8 graph |
| Threads need cross-origin isolation | Without COOP and COEP headers the page runs on one thread, about 1 s per step |
| One more download | ONNX Runtime Web's engine is 14.2 MB, shared and cacheable |
| Offline evaluation | Recall on recorded pages, not task success. Tokens are estimated at 4 characters each. Steps whose clicked element is missing from the HTML (4-7%) count as misses |
