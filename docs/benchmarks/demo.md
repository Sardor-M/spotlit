# Demo results

The interactive demo plays recorded Mind2Web tasks through the released model in the
browser: per step it shows the pipeline stages with live counts and timings, the page's
elements re-sorting from stage 1's order to the final order, the top elements the agent
would read, and tokens saved. It also ranks any pasted HTML.

## What it runs

- **Tasks.** 8 complete tasks, 36 steps, from 8 websites in the dev split (Mind2Web train
  split websites never trained on; CC BY 4.0). No test-split data is shown.
- **Runtime.** The release files, the dependency-free stage 1 runtime and WordPiece
  tokenizer, and ONNX Runtime Web 1.30.0 on one WebAssembly thread (hosted pages are not
  cross-origin isolated).

## Results

| Measure | Value |
| :--- | :--- |
| Clicked element in the top 50 | 29 of 36 steps |
| Agreement with the Python pipeline on top-50 membership | 36 of 36 steps |
| Time per page step, one thread | about 1 s |
| Time per page step, eight threads (bench page) | about 0.2 s |
| Tokens per step, top 50 vs cleaned HTML | median 27x fewer (7x to 154x by page) |

Two of the eight tasks are shown because the model struggles on them: on a travel site the
clicked element misses the top 50 on four of five steps, and on a car retailer on two of six.
A first version of the demo with 30 single steps from 10 websites agreed with Python on
30/30 and kept the clicked element in the top 50 on 24.

## Pasted HTML

On a sample restaurant page with the goal "Book a table for two people tomorrow at 7 pm",
the top three are the name field, the date field and the time selector.
