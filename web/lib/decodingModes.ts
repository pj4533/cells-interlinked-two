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

export const DECODING_MODE_DESCRIPTIONS: Record<DecodingMode, string> = {
  "per-token":
    "Per-token — every output position is NLA-decoded. Slowest, fullest signal. ~10s per position; an 80-token answer takes ~13 min.",
  "every-3rd":
    "Every 3rd — captures the narrative shape at one third of the cost. Smooth read; loses moment-to-moment detail.",
  "every-5th":
    "Every 5th — recommended for batch runs. ~5× faster than per-token while keeping enough positions to see drift across an answer.",
  "key-points":
    "Key points — five strategic positions only (first, ~25%, ~50%, ~75%, last). ~30-60s per probe. Coarse but very fast.",
};

export function isDecodingMode(s: string): s is DecodingMode {
  return (DECODING_MODES as string[]).includes(s);
}
