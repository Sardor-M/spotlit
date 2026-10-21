# browser-agent

Test spotlit inside real agents. Claude Code or Codex drives a Chrome window through a small MCP
server, once seeing every element on the page and once seeing only spotlit's top 50, and you
compare success, tokens and time.

## How it works

`mcp.mjs` is an MCP server with seven browser tools: `open`, `elements`, `click`, `type`,
`select`, `press`, `back`. Its two modes differ only in what `elements` returns:

| Mode | `elements` returns | Stands for |
| :--- | :--- | :--- |
| `--mode full` | every visible element in page order, as `[id] description` | an agent without spotlit |
| `--mode spotlit` | the top k (default 50) for a `goal` the agent passes, best first | the same agent with spotlit |

Both modes collect elements with spotlit's own `collectCandidates` from `web/src/spotlit.js`,
so the descriptions are identical; only visible elements count, up to 2,000 per page. Ranking
runs in Node on ONNX Runtime Web's WebAssembly backend, the engine the browser uses, on one
thread (0.2 to 1.4 s per page; Chrome with 8 threads is about 5x faster). On the 40 bench steps
it agrees with the Python pipeline on whether the gold element makes the top 50 in 40 of 40.

Every tool call is logged to a JSONL file: tool, arguments, page URL, and the size of what the
model was shown.

## Setup

```sh
cd browser-agent && npm install
```

Needs Node 22 or newer, Google Chrome (set `CHROME` for another binary), and the agent CLIs
`claude` and `codex`, logged in. `CLAUDE_BIN` and `CODEX_BIN` point at a specific binary, for
example when `claude` on your PATH is a wrapper that adds hooks.

**Every command below runs from inside `browser-agent/`.** From the repo root, put
`browser-agent/` before the script name (`node browser-agent/run.mjs ...`). Run one command at a
time: pasting a whole block starts each line as soon as the one before it ends.

## 1. Side by side, by hand

Two terminals:

```sh
node agent.mjs claude full    --side left  --task books-price
node agent.mjs claude spotlit --side right --task books-price
```

Each opens an interactive Claude Code session that has only the browser tools, a Chrome window
on its half of the screen, and the task already sent. Use `codex` in place of `claude` for Codex.
Leave out `--task` to type your own task; tell the agent to use only the browser tools and to
open nothing but the start page, or it will skip the page by typing URLs. When you quit, the
launcher prints how many browser calls were made and how many characters of element lists the
model received. `/context` in Claude Code and the status line in Codex show the token totals.

### Demo tasks for a side-by-side recording

`demo-tasks.json` holds seven longer tasks: several pages each, search boxes, pagination, a form,
and answers read from tables. Run the same id in both terminals:

```sh
node agent.mjs claude full    --side left  --task demo-wiki-mountains --model sonnet
node agent.mjs claude spotlit --side right --task demo-wiki-mountains --model sonnet
```

Every pair, ready to copy. The first line of each pair goes in the left terminal, the second in
the right; quit both sessions (Ctrl+C twice) before starting the next pair.

```sh
node agent.mjs claude full    --side left  --task demo-wiki-tallest --model sonnet
node agent.mjs claude spotlit --side right --task demo-wiki-tallest --model sonnet

node agent.mjs claude full    --side left  --task demo-wiki-mountains --model sonnet
node agent.mjs claude spotlit --side right --task demo-wiki-mountains --model sonnet

node agent.mjs claude full    --side left  --task demo-books-cheapest --model sonnet
node agent.mjs claude spotlit --side right --task demo-books-cheapest --model sonnet

node agent.mjs claude full    --side left  --task demo-books-upc --model sonnet
node agent.mjs claude spotlit --side right --task demo-books-upc --model sonnet

node agent.mjs claude full    --side left  --task demo-quotes-humor --model sonnet
node agent.mjs claude spotlit --side right --task demo-quotes-humor --model sonnet

node agent.mjs claude full    --side left  --task demo-hockey-losses --model sonnet
node agent.mjs claude spotlit --side right --task demo-hockey-losses --model sonnet

node agent.mjs claude full    --side left  --task demo-pydocs-deque --model sonnet
node agent.mjs claude spotlit --side right --task demo-pydocs-deque --model sonnet
```

If the agent prints tool calls as text (`<invoke name="open">`) or a session ends with "0 browser
tool calls", the browser tools were not loaded. The launcher guards against the two causes seen
so far:

- Interactive Claude Code can send the first prompt about half a second after it starts, before
  any tool is registered, so the model starts with none. The launcher adds a start-up hook
  (`--settings`) that waits until the browser server has sent its tool list, up to 15 seconds.
  Claude Code keeps only the last `--settings`, so a wrapper's own hooks, such as cmux's, are
  skipped for these sessions.
- With Claude in Chrome turned on by default, the session is told to use its
  `mcp__claude-in-chrome__*` tools instead. The launcher passes `--no-chrome`.

Both need a recent Claude Code; `claude --help` should list `--no-chrome`.

Keep both Chrome windows open and the Mac awake while a pair runs. If a window is closed, the
next browser call says so and `open` starts a new window, but the run is no longer comparable.

| Task id | Start page | What it exercises | Correct answer |
| :--- | :--- | :--- | :--- |
| `demo-wiki-tallest` | Wikipedia, Japan | Two hops across three huge pages (2,000+ elements each) | Tokyo Skytree, 634 m |
| `demo-wiki-mountains` | Wikipedia, Main Page | Three searches, reading three infoboxes | K2 8,611 m; Kangchenjunga 8,586 m; Lhotse 8,516 m |
| `demo-books-cheapest` | books.toscrape.com | Category menu, comparing 11 prices, a product page | The Road to Little Dribbling, £23.21, 3 in stock |
| `demo-books-upc` | books.toscrape.com | Four clicks on "next", then a product table | UPC 0fa6dceead7ce47a |
| `demo-quotes-humor` | quotes.toscrape.com | A tag page over two pages, counting | 12 quotes; Jane Austen appears twice |
| `demo-hockey-losses` | scrapethissite.com forms | Typing into a search form, reading a 21-row table | 1990, 38 losses |
| `demo-pydocs-deque` | docs.python.org library index | A big index, then a 2,000+ element docs page | 3.1 |

Each answer was checked on the live site on 2026-10-19. The UPC and hockey answers cannot be
known in advance, so they show best that the agent read the page. The mountain heights and the
Python version are well known, and in the check run below Claude wrote known heights into its
spotlit goals (`"elevation metres 8,586 m"`) before reaching that page.

The agent receives the task wrapped in the same rules for both modes. For `demo-hockey-losses`:

```text
You control a Chrome browser through the browser tools: open, elements, click, type, select, press, back.
Task: Search the hockey team statistics for "Detroit Red Wings". In the results, find the season with the most losses and report its year and its number of losses.
Start by opening https://www.scrapethissite.com/pages/forms/
Use open only for that start page. After it, move around the site only by clicking, typing, selecting and pressing keys on the page.
Use only the browser tools. Do not run shell commands or search the web.
Element ids change whenever the page changes, so call elements again after every action.
When the task is done, end your reply with one line: ANSWER: <the answer, or "done">
```

For a task of your own, run both commands without `--task` and paste the same text into both
sessions, with your task and start page in the second and third lines. To make it repeatable,
add it to `demo-tasks.json` instead and pass its id.

**Check run, 2026-10-20.** Claude Code on Sonnet, headless, one run per task and mode, both modes
at the same time (`run.mjs --tasks demo-tasks.json`). Raw logs are in `runs/agent/demo-check2-*`.

| Task | Full: input tokens | spotlit: input tokens | Full: time, calls | spotlit: time, calls |
| :--- | ---: | ---: | ---: | ---: |
| demo-wiki-tallest | 826,382 | 46,230 | 29 s, 8 | 24 s, 6 |
| demo-wiki-mountains | 1,248,771 | 78,413 | 28 s, 8 | 32 s, 9 |
| demo-books-cheapest | 66,165 | 69,388 | 17 s, 6 | 21 s, 8 |
| demo-books-upc | 240,949 | 83,631 | 33 s, 12 | 40 s, 13 |
| demo-quotes-humor | 64,414 | 38,273 | 15 s, 6 | 18 s, 6 |
| demo-hockey-losses | 60,708 | 50,459 | 13 s, 4 | 16 s, 5 |
| demo-pydocs-deque | 192,995 | 61,393 | 14 s, 4 | 22 s, 8 |
| **All seven** | **2,700,384 ($3.55), 7 of 7 passed** | **427,787 ($0.48), 7 of 7 passed** | 149 s, 48 calls | 173 s, 55 calls |

The first check run, on 2026-10-19, had two problems since fixed. Clicks on links that wrap onto
two lines landed on the heading around them (13 of 31 book-title links at the default width), and
the spotlit agent could ask for as few as 3 elements, miss, and ask again. Totals then were 306 s
and 65 calls for spotlit against 236 s and 49 calls for full.

What to look for in the recording:

- **The gap grows with page size and step count.** On Wikipedia, spotlit read 16 to 18 times fewer
  tokens; on the Python docs and the 13-step `demo-books-upc`, about 3 times fewer. On short tasks
  over small pages it read about the same.
- **The full side is a little faster.** Wall time is mostly model turns of 2 to 4 seconds each, not
  reading: a 20,000-character list costs the model well under a second, and repeated lists are
  cached. The spotlit side takes a few more turns, because the agent sometimes asks again with a
  different goal or a larger list, and each ranking adds 0.1 to 1.5 s in Node on one thread.
- **The agent asks for more when it needs to.** It gets 50 elements by default and asks for 100 to
  400 when reading a product grid, a quote list or a table.
- **Full mode piles up context.** Every page listing stays in the conversation and is re-read on
  each later step; `demo-wiki-mountains` reached 1.25 million input tokens over 8 calls.

## 2. Batch runs on live sites

Each of these is a separate run; start the next when the previous one has printed its summary.

Page sizes and spotlit's rank of each first target, with no LLM and no cost:

```sh
node probe.mjs
node probe.mjs --tasks demo-tasks.json
```

The five short tasks, Claude in both modes, one after the other, without windows (about $1.40 at
API prices):

```sh
node run.mjs --agent claude --model sonnet --headless
```

The seven demo tasks the same way (about $4):

```sh
node run.mjs --agent claude --model sonnet --headless --tasks demo-tasks.json
```

Both modes at once in two visible windows, here for one demo task:

```sh
node run.mjs --agent claude --model sonnet --parallel --tasks demo-tasks.json --only demo-wiki-mountains
```

Codex, once its usage limit allows:

```sh
node run.mjs --agent codex --only wiki-skytree,books-price
```

Tasks live in `tasks.json`, and the longer demo tasks in `demo-tasks.json` (pass
`--tasks demo-tasks.json`). Each has a start URL, the task, and a check: `url` the
last page must contain, and `answer`, a string or a list of strings the final reply must all
contain. `probe` is the text of the first target and `goal` the first step's goal, which
`probe.mjs` ranks with; agents pass short step goals like these, and ranking with a whole
multi-step task pushes the first target much lower. Each run writes `runs/agent/<time>/` with
`results.jsonl`, `summary.md`, and one folder per run holding the agent's raw output and the
browser tool log. `--repeat 3` reruns everything, because live sites and models vary.

## 3. Offline replay on Mind2Web

```sh
node replay.mjs --agent claude --model sonnet --n 30
```

Each sampled step is asked twice in fresh sessions with the same model, once with every element
and once with spotlit's top 50, and scored the Mind2Web way: the reply must name the exact gold
element. It reads only `data/mind2web/dev.jsonl`; the test splits are never sent to an API.

## Pilot results, 2026-10-18

One run each, so treat these as a first look, not a benchmark. Raw logs are in `runs/agent/pilot-*`.

**Live tasks, Claude Code on Sonnet.** Both modes finished every task. spotlit cut the tokens the
agent read by 6.6x and the cost by 11x. Time stayed the same, because model calls dominate it.

| Task | Elements on the start page | Full: input tokens | spotlit: input tokens |
| :--- | ---: | ---: | ---: |
| wiki-skytree | 2,000 (capped) | 171,284 | 18,906 |
| wiki-search | 707 | 199,390 | 29,126 |
| books-price | 241 | 105,920 | 25,030 |
| quotes-love | 117 | 52,250 | 32,127 |
| pydocs-groupby | 891 | 327,711 | 25,057 |
| **All five** | | **856,555 ($1.25), 5 of 5 passed** | **130,246 ($0.11), 5 of 5 passed** |

**Live tasks, Codex on its default model.** The account reached its usage limit after three
tasks. Before that, both modes passed wiki-skytree (136,148 vs 68,156 tokens) and quotes-love
(110,687 vs 93,925). Codex re-reads a larger fixed prompt on every call, so the ratio is smaller.

**Mind2Web dev replay, 30 random steps (seed 1).** One question per step: which element is next.

| Agent | List shown | Right element | Gold element in the list | Input tokens |
| :--- | :--- | ---: | ---: | ---: |
| Claude Code, Sonnet | every element | 15 of 30 | 30 of 30 | 964,166 ($3.57) |
| Claude Code, Sonnet | spotlit top 50 | 13 of 30 | 26 of 30 | 76,740 ($0.32) |
| Codex, default model | every element | 12 of 23 | 23 of 23 | 560,558 |
| Codex, default model | spotlit top 50 | 9 of 23 | 20 of 23 | 210,525 |

Codex covers the 23 steps answered before its usage limit. Step by step, Claude got 9 right
either way, 6 only with the full list, and 4 only with spotlit's list. The gold element was
outside spotlit's top 50 on 4 steps, and the full list got 3 of those. So spotlit costs a little
accuracy, mostly when its recall misses (87% at 50 here, 73% across all dev steps), and gains
some back by showing the model a shorter list. With 30 steps, a difference of 2 is within noise;
run `--n 200` before drawing a conclusion.

## Keeping the comparison fair

- Same prompt, same tools, same element descriptions in both modes; only the `elements` output
  differs.
- Claude Code replaces any tool result over about 50,000 characters with a file path and a 2 KB
  preview. Without a fix, the full mode goes blind on large pages: in the first pilot the agent
  clicked ids it could not see. `elements` therefore sets
  `_meta["anthropic/maxResultSizeChars"]` to 600,000, and the runners set
  `MAX_MCP_OUTPUT_TOKENS=200000`, so full lists arrive inline.
- Claude Code runs with `--tools ""` and `--no-chrome`, and Codex with its own browser, plugins, shell and web search
  turned off, so neither can bypass the server under test.
- Token totals include each CLI's own system prompt and tool definitions, which are re-read on
  every model call (about 6,000 tokens for Claude Code, mostly cached, and 7,000 or more for
  Codex). That fixed cost narrows the ratio on small pages.
- Pages with more than 2,000 visible elements are cut at 2,000 in both modes, and the listing
  says so.
- The replay sorts the full list by element id, which is page order, because Mind2Web candidate
  lists put the gold element first.
- Mind2Web labels are strict. When the gold element is a text-less `svg` icon, picking the "Size"
  heading next to it scores as wrong. Both arms are scored the same way, so this is noise, not bias.
- Full-mode runs on big pages read 100,000 to 300,000 tokens each. A free Codex plan reached its
  monthly usage limit during the pilot.
