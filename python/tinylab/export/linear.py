"""Quantize the linear student to per-row INT8 and write the `.tlab` file the runtimes load. See ADR-0009."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from ..data.make_encoder_goldens import CASES
from ..encoder import EncoderSpec, features
from ..eval.student import checkpoint_stem

MAGIC = b"TLAB"
FORMAT_VERSION = 1
BUDGET_BYTES = 2_000_000
FIXED_HEADER = struct.Struct("<4sIIIIII")


def quantize(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row INT8: row b is approximately q[b] * scales[b]."""
    scales = (np.abs(table).max(axis=1) / 127).astype(np.float32)
    divisor = np.where(scales > 0, scales, 1.0).astype(np.float64)
    q = np.clip(np.rint(table / divisor[:, None]), -127, 127).astype(np.int8)
    return q, scales


def encode(q: np.ndarray, scales: np.ndarray, spec: EncoderSpec) -> bytes:
    sizes = spec.ngram_sizes
    header = FIXED_HEADER.pack(
        MAGIC, FORMAT_VERSION, spec.version, spec.num_buckets, q.shape[1], spec.max_input_bytes, len(sizes)
    ) + struct.pack(f"<{len(sizes)}I", *sizes)
    return header + scales.astype("<f4").tobytes() + q.tobytes()


def decode(blob: bytes) -> tuple[EncoderSpec, np.ndarray, np.ndarray]:
    magic, fmt, version, buckets, dims, max_bytes, n_sizes = FIXED_HEADER.unpack_from(blob)
    if magic != MAGIC or fmt != FORMAT_VERSION:
        raise ValueError(f"not a version {FORMAT_VERSION} .tlab file")
    sizes = struct.unpack_from(f"<{n_sizes}I", blob, FIXED_HEADER.size)
    start = FIXED_HEADER.size + 4 * n_sizes
    scales = np.frombuffer(blob, dtype="<f4", count=buckets, offset=start)
    table_start = start + 4 * buckets
    q = np.frombuffer(blob, dtype=np.int8, count=buckets * dims, offset=table_start).reshape(buckets, dims)
    spec = EncoderSpec(version=version, num_buckets=buckets, ngram_sizes=tuple(sizes), max_input_bytes=max_bytes)
    return spec, scales, q


def embed(text: str, spec: EncoderSpec, scales: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Reference embedding from the quantized weights, float64 throughout."""
    pairs = features(text, spec)
    out = np.zeros(q.shape[1])
    for bucket, count in pairs:
        out += count * float(scales[bucket]) * q[bucket].astype(np.float64)
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def golden_texts() -> list[tuple[str, str]]:
    return CASES + [
        ("lumen-like-heading", "Reciprocal Rank Fusion"),
        ("lumen-like-question", "how does the daemon admit ingest requests"),
        ("identifier-heavy", "LUMEN_ENGINE_SEARCH=rust store/schema.ts:78-96 ADR-083"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-export", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/student-linear-6144x320-design"))
    parser.add_argument("--out", type=Path, default=Path("dist/tinylab-student-v1"))
    args = parser.parse_args(argv)

    import mlx.core as mx

    stem = checkpoint_stem(args.checkpoint)
    meta = json.loads(stem.with_name(stem.name + ".json").read_text(encoding="utf-8"))
    if meta["config"].get("hidden", 0) or meta["config"].get("kind"):
        raise ValueError("tinylab-export handles the linear student only (ADR-0009)")
    table = np.array(mx.load(str(stem.with_name(stem.name + ".safetensors")))["table.weight"], dtype=np.float64)
    spec = EncoderSpec(num_buckets=table.shape[0])

    q, scales = quantize(table)
    blob = encode(q, scales, spec)
    if len(blob) > BUDGET_BYTES:
        raise SystemExit(f"{len(blob):,} bytes is over the {BUDGET_BYTES:,}-byte budget (ADR-0008)")
    spec_back, scales_back, q_back = decode(blob)
    assert spec_back == spec and np.array_equal(scales_back, scales) and np.array_equal(q_back, q)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "student.tlab").write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    error = np.abs(q.astype(np.float64) * scales[:, None] - table).max()
    manifest = {
        "format": "tinylab-linear-int8",
        "format_version": FORMAT_VERSION,
        "file": "student.tlab",
        "bytes": len(blob),
        "sha256": digest,
        "encoder": {"version": spec.version, "num_buckets": spec.num_buckets,
                    "ngram_sizes": list(spec.ngram_sizes), "max_input_bytes": spec.max_input_bytes},
        "dims": int(q.shape[1]),
        "params": int(q.size),
        "quantization": "symmetric per-row int8, float32 scales",
        "max_abs_weight_error": float(error),
        "source_checkpoint": str(stem),
    }
    (args.out / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    goldens = [
        {"name": name, "text": text, "vector": [round(float(x), 8) for x in embed(text, spec, scales, q)]}
        for name, text in golden_texts()
    ]
    (args.out / "goldens.json").write_text(json.dumps({"sha256": digest, "cases": goldens}), encoding="utf-8")

    dequantized = stem.with_name(stem.name + "-int8")
    weights = mx.array((q * scales[:, None]).astype(np.float32))
    mx.save_safetensors(str(dequantized) + ".safetensors", {"table.weight": weights})
    dequantized.with_name(dequantized.name + ".json").write_text(
        json.dumps({"config": meta["config"], "dequantized_from": str(args.out / "student.tlab")}, indent=2),
        encoding="utf-8",
    )
    print(
        f"wrote {args.out}/student.tlab  {len(blob):,} bytes (budget {BUDGET_BYTES:,})  sha256 {digest[:16]}"
        f"  max weight error {error:.2e}\n  scoring copy {dequantized}.safetensors",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
