"""Export the stage 2 re-ranker to ONNX (float32 and INT8) for ONNX Runtime Web, with goldens. See ADR-0014."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .rank import load_steps, query_of
from .rerank import MAX_TOKENS

GOLDEN_PAIRS = 400
INPUTS = ("input_ids", "attention_mask", "token_type_ids")
OPSET = 17


class OnnxEncoder:
    """Same interface as CrossEncoder, scored by ONNX Runtime on the CPU."""

    def __init__(self, model_path: Path, tokenizer):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizer

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), 512):
            window = texts[start : start + 512]
            batch = self.tokenizer([query] * len(window), window, truncation=True, max_length=MAX_TOKENS,
                                   padding="longest", return_tensors="np")
            feeds = {name: batch[name].astype(np.int64) for name in INPUTS}
            out.append(self.session.run(["score"], feeds)[0])
        return np.concatenate(out) if out else np.zeros(0)


def export(directory: Path, out: Path) -> tuple[Path, Path]:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(directory, num_labels=1).eval()

    class Scorer(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return out.logits[:, 0]

    sample = torch.ones((2, 16), dtype=torch.long)
    fp32 = out / "reranker.fp32.onnx"
    torch.onnx.export(
        Scorer(model), (sample, sample, torch.zeros_like(sample)), str(fp32), dynamo=False, opset_version=OPSET,
        input_names=list(INPUTS), output_names=["score"],
        dynamic_axes={name: {0: "pairs", 1: "tokens"} for name in INPUTS} | {"score": {0: "pairs"}},
    )
    int8 = out / "reranker.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return fp32, int8


EDGE_TEXTS = [
    "Café naïve résumé Ångström", "İstanbul TITLE Iıİi straße STRASSE", "日本語のテキスト検索 and 한국어 문서",
    "emoji 🔍🚀 👨‍👩‍👧 ok", "tab\there\nnew line\r\nend", "zero\u200bwidth\u200djoiner\ufeffbom",
    "control\u0007bell\u0000nul\ufffdrepl",
    "symbols © ® € £ ¥ § ¶ † ‡ • … ™ ± × ÷",
    "quotes “curly” ‘single’ «guillemets» — dash – en", "x" * 120, "C++ C# .NET node.js e-mail",
    "price: $1,299.99 (20% off) [buy] {now}", "Ελληνικά Русский العربية עברית हिन्दी",
    "", "ﬁ ligature ﬂ ½ ²",
]
GRID = [1, 2, 61, 62, 63, 64, 70, 124, 125, 126, 127, 128, 129, 200]


def write_goldens(steps: list[dict], tokenizer, encoder: OnnxEncoder, path: Path) -> None:
    """Token ids (and INT8 scores for real pairs) for the JS tokenizer and runtime parity tests."""
    cases = []
    for step in steps:
        for _, text in step["candidates"][: max(1, GOLDEN_PAIRS // len(steps))]:
            query = query_of(step)
            cases.append({"query": query, "element": text, "score": float(encoder.score(query, [text])[0])})
    cases += [{"query": a, "element": b} for a in EDGE_TEXTS for b in ("search button", *EDGE_TEXTS[::4])]
    cases += [{"query": " ".join(["a"] * n1), "element": " ".join(["b"] * n2)} for n1 in GRID for n2 in GRID]
    for case in cases:
        encoded = tokenizer(case["query"], case["element"], truncation=True, max_length=MAX_TOKENS)
        case["input_ids"], case["token_type_ids"] = encoded["input_ids"], encoded["token_type_ids"]
    path.write_text(json.dumps(cases), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-export-reranker", description=__doc__)
    parser.add_argument("--reranker", type=Path, required=True, help="fine-tuned checkpoint directory")
    parser.add_argument("--out", type=Path, help="default: dist/<reranker name>")
    parser.add_argument("--dev", type=Path, default=Path("data/mind2web/dev.jsonl"))
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    out = args.out or Path("dist") / args.reranker.name
    out.mkdir(parents=True, exist_ok=True)
    fp32, int8 = export(args.reranker, out)
    tokenizer = AutoTokenizer.from_pretrained(args.reranker)
    tokenizer.save_pretrained(out)
    vocab = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    (out / "vocab.txt").write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")
    config = json.loads((args.reranker / "tokenizer_config.json").read_text(encoding="utf-8"))
    (out / "wordpiece.json").write_text(json.dumps({
        "do_lower_case": config.get("do_lower_case", True), "strip_accents": config.get("strip_accents"),
        "tokenize_chinese_chars": config.get("tokenize_chinese_chars", True), "max_length": MAX_TOKENS,
        "truncation": "longest_first",
    }, indent=2), encoding="utf-8")

    steps = load_steps(args.dev)[:: 100]
    write_goldens(steps, tokenizer, OnnxEncoder(int8, tokenizer), out / "goldens.json")
    sizes = {path.name: path.stat().st_size for path in (fp32, int8, out / "vocab.txt")}
    print(f"wrote {out}: " + ", ".join(f"{name} {size / 1e6:.2f} MB" for name, size in sizes.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
