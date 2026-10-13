// Tokenizer-free text embeddings from a .tlab file, no dependencies. Format: python/tinylab/export/README.md.

const SPACE = 0x20;
const BASIS_HI = 0xcbf29ce4;
const BASIS_LO = 0x84222325;
const PRIME_LO = 0x1b3;
const TWO_32 = 0x100000000;
const FIXED_HEADER_BYTES = 28;

let hashHi = 0;
let hashLo = 0;

// (hashHi:hashLo) *= 0x100000001b3 mod 2^64, in 32-bit halves.
function multiplyByPrime() {
  const low = hashLo * PRIME_LO;
  const carry = Math.floor(low / TWO_32);
  hashHi = (Math.imul(hashHi, PRIME_LO) + carry + (hashLo & 0xffffff) * 256) >>> 0;
  hashLo = low >>> 0;
}

function hashNgram(bytes, start, size) {
  hashHi = BASIS_HI;
  hashLo = BASIS_LO;
  for (let index = 0; index < size; index++) {
    hashLo = (hashLo ^ bytes[start + index]) >>> 0;
    multiplyByPrime();
  }
  hashLo = (hashLo ^ size) >>> 0;
  multiplyByPrime();
}

/** UTF-8 bytes, ASCII-lowercased, non-alphanumeric collapsed to single spaces (encoder v1). */
export function normalize(text, maxInputBytes) {
  const raw = new TextEncoder().encode(text);
  const out = new Uint8Array(raw.length);
  let length = 0;
  let previousWasSpace = true;
  for (const byte of raw) {
    let current = SPACE;
    if (byte >= 0x41 && byte <= 0x5a) current = byte + 0x20;
    else if ((byte >= 0x61 && byte <= 0x7a) || (byte >= 0x30 && byte <= 0x39) || byte >= 0x80) current = byte;
    if (current === SPACE) {
      if (previousWasSpace) continue;
      previousWasSpace = true;
    } else {
      previousWasSpace = false;
    }
    out[length++] = current;
  }
  if (length > 0 && out[length - 1] === SPACE) length--;
  return out.subarray(0, Math.min(length, maxInputBytes));
}

/** Sorted [bucket, signed count] pairs, identical to tinylab.encoder.features. */
export function features(text, spec) {
  const counts = new Map();
  forEachBucket(text, spec, (bucket, sign) => counts.set(bucket, (counts.get(bucket) ?? 0) + sign));
  return [...counts].filter(([, count]) => count !== 0).sort((a, b) => a[0] - b[0]);
}

function forEachBucket(text, spec, visit) {
  const normalized = normalize(text, spec.maxInputBytes);
  if (normalized.length === 0) return;
  const padded = new Uint8Array(normalized.length + 2);
  padded[0] = SPACE;
  padded.set(normalized, 1);
  padded[padded.length - 1] = SPACE;
  const highFactor = TWO_32 % spec.numBuckets;
  for (const size of spec.ngramSizes) {
    for (let start = 0; start + size <= padded.length; start++) {
      hashNgram(padded, start, size);
      const bucket = ((hashHi % spec.numBuckets) * highFactor + (hashLo % spec.numBuckets)) % spec.numBuckets;
      visit(bucket, hashHi >>> 31 ? -1 : 1);
    }
  }
}

export class Student {
  constructor(spec, dims, scales, table) {
    this.spec = spec;
    this.dims = dims;
    this.scales = scales;
    this.table = table;
    this.counts = new Int32Array(spec.numBuckets);
    this.seen = new Uint8Array(spec.numBuckets);
    this.touched = new Int32Array(spec.numBuckets);
  }

  /** Parse a .tlab file; the scales and table are views into `buffer`, not copies. */
  static fromBuffer(buffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
    if (magic !== "TLAB" || view.getUint32(4, true) !== 1) throw new Error("not a version 1 .tlab file");
    const numBuckets = view.getUint32(12, true);
    const dims = view.getUint32(16, true);
    const maxInputBytes = view.getUint32(20, true);
    const sizeCount = view.getUint32(24, true);
    const ngramSizes = Array.from({ length: sizeCount }, (_, i) => view.getUint32(FIXED_HEADER_BYTES + 4 * i, true));
    const scalesStart = FIXED_HEADER_BYTES + 4 * sizeCount;
    const scales = new Float32Array(buffer, scalesStart, numBuckets);
    const table = new Int8Array(buffer, scalesStart + 4 * numBuckets, numBuckets * dims);
    return new Student({ numBuckets, ngramSizes, maxInputBytes }, dims, scales, table);
  }

  /** L2-normalized embedding; cosine similarity is the dot product of two of these. */
  embed(text) {
    const { counts, seen, touched } = this;
    let present = 0;
    forEachBucket(text, this.spec, (bucket, sign) => {
      if (!seen[bucket]) {
        seen[bucket] = 1;
        touched[present++] = bucket;
      }
      counts[bucket] += sign;
    });
    const out = new Float64Array(this.dims);
    for (let i = 0; i < present; i++) {
      const bucket = touched[i];
      const weight = counts[bucket] * this.scales[bucket];
      counts[bucket] = 0;
      seen[bucket] = 0;
      if (weight === 0) continue;
      const row = bucket * this.dims;
      for (let d = 0; d < this.dims; d++) out[d] += weight * this.table[row + d];
    }
    let norm = 0;
    for (let d = 0; d < this.dims; d++) norm += out[d] * out[d];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let d = 0; d < this.dims; d++) out[d] /= norm;
    return out;
  }
}

export function cosine(a, b) {
  let dot = 0;
  for (let d = 0; d < a.length; d++) dot += a[d] * b[d];
  return dot;
}
