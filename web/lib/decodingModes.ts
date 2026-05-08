export type DecodingMode =
  | "per-token"
  | "every-3rd"
  | "every-5th"
  | "key-points";

export const DECODING_MODES: DecodingMode[] = [
  "per-token",
  "every-3rd",
  "every-5th",
  "key-points",
];

export const DECODING_MODE_LABELS: Record<DecodingMode, string> = {
  "per-token": "PER-TOKEN",
  "every-3rd": "EVERY 3RD",
  "every-5th": "EVERY 5TH",
  "key-points": "KEY POINTS",
};

export const DECODING_MODE_SAMPLED_DESCRIPTIONS: Record<DecodingMode, string> = {
  "per-token":
    "Per-token — every output position is NLA-decoded individually. Each row reads what one specific token's activation says. Slowest, fullest signal.",
  "every-3rd":
    "Every 3rd, sampled — decodes positions 0, 3, 6, 9, … one at a time. The activations at the skipped positions are discarded. Each row is still a per-token read; you just have fewer of them.",
  "every-5th":
    "Every 5th, sampled — decodes positions 0, 5, 10, 15, … one at a time. Skipped positions are discarded. Per-token reads, ~5× fewer rows.",
  "key-points":
    "Key points, sampled — decodes 5 strategic positions (first, ~25/50/75%, last) one at a time. Per-token reads at five hand-picked moments.",
};

export const DECODING_MODE_POOLED_DESCRIPTIONS: Record<DecodingMode, string> = {
  "per-token":
    "Per-token — pooled has no effect at stride 1. Every position is its own row.",
  "every-3rd":
    "Every 3rd, pooled — for each window of 3 (positions 0–2, 3–5, 6–8, …), the activations are mean-pooled into one vector and decoded once. Each row is a phrase-level read covering 3 tokens.",
  "every-5th":
    "Every 5th, pooled — for each window of 5 (positions 0–4, 5–9, …), the activations are mean-pooled and decoded once. Each row is a phrase-level read covering 5 tokens.",
  "key-points":
    "Key points, pooled — each of the 5 key positions becomes a ±2 window (5 tokens centered on the key point). Mean-pooled and decoded as one phrase-level read each.",
};

export function modeDescription(mode: DecodingMode, pooled: boolean): string {
  return pooled
    ? DECODING_MODE_POOLED_DESCRIPTIONS[mode]
    : DECODING_MODE_SAMPLED_DESCRIPTIONS[mode];
}

export function isDecodingMode(s: string): s is DecodingMode {
  return (DECODING_MODES as string[]).includes(s);
}
