// WordPiece parity with the Python tokenizer: node web/test/wordpiece-parity.mjs [exported re-ranker dir]
import { readFileSync } from "node:fs";
import { isAbsolute } from "node:path";
import { pathToFileURL } from "node:url";
import { WordPiece } from "../src/wordpiece.js";

const arg = process.argv[2] ?? "dist/reranker-bert_uncased_L-2_H-128_A-2-v1";
const dir = isAbsolute(arg) ? pathToFileURL(`${arg}/`) : new URL(`../../${arg}/`, import.meta.url);
const tokenizer = new WordPiece(readFileSync(new URL("vocab.txt", dir), "utf8"));
const cases = JSON.parse(readFileSync(new URL("goldens.json", dir), "utf8"));
let failures = 0, truncated = 0;
for (const c of cases) {
  const got = tokenizer.encodePair(c.query, c.element);
  if (c.input_ids.length === 128) truncated++;
  const same = got.inputIds.join() === c.input_ids.join() && got.tokenTypeIds.join() === c.token_type_ids.join();
  if (!same && failures++ < 3) {
    const at = got.inputIds.findIndex((id, i) => id !== c.input_ids[i]);
    console.log(`  FAIL at token ${at}: ${JSON.stringify(c.element).slice(0, 80)}`);
  }
}
console.log(`wordpiece parity: ${cases.length - failures}/${cases.length} pairs match (${truncated} truncated to 128)`);
process.exit(failures ? 1 : 0);
