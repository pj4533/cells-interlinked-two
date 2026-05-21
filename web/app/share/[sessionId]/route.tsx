/** Server-rendered share card covering one or more chat turns.
 *
 *  GET /share/{sessionId}?turns=0,2,5&format=polaroid|dossier
 *                                                   → image/png
 *
 *  `turns` is a comma-separated list of turn_idx values. If absent,
 *  every turn in the session is included. Order is honoured: the
 *  selected turn blocks render in the order the operator passed
 *  them. The canvas height grows to fit content; the full operator
 *  prompt is rendered without truncation.
 *
 *  Two layouts:
 *   - polaroid (1080 × ~variable) — imagery-forward; each selected
 *     turn becomes a "polaroid panel" stacked vertically.
 *   - dossier  (1080 × ~variable) — text-forward; each selected turn
 *     becomes a two-column block stacked vertically.
 *
 *  No URLs are stamped on the artifact. Only the "cells interlinked"
 *  wordmark + α / variant / model footer.
 */

import { ImageResponse } from "next/og";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// ── Types matching backend SessionView / TurnView ────────────────

interface TurnView {
  turn_idx: number;
  user_text: string;
  raw_text: string;
  ablated_text: string;
  raw_stopped_reason: string;
  ablated_stopped_reason: string;
  alpha: number;
  raw_image_url?: string;
  ablated_image_url?: string;
}
interface SessionView {
  session_id: string;
  alpha: number;
  direction_variant: string;
  created_at: number;
  turns: TurnView[];
}

// ── Palette (matches web/app/globals.css :root tokens) ──────────

const BG = "#0a0d10";
const BG_SOFT = "#11151a";
const AMBER = "#e8c382";
const AMBER_DIM = "#8a7349";
const CYAN = "#5ee5e5";
const CYAN_DIM = "#3a8d8d";
const TEXT_DIM = "#6f7a83";
const RULE = "#2a3038";

// ── Layout constants — used both for the JSX and for the height
//    estimate. Keep these in sync if you change padding / sizes. ──

const POLAROID_WIDTH = 1080;
const POLAROID_IMAGE = 380;
const POLAROID_TEXT_BLOCK_LINES_AT_240_CHARS = 9; // ~9 lines at 18px italic in a 380-wide column

const DOSSIER_WIDTH = 1080;
const DOSSIER_TEXT_FONT = 20;
const DOSSIER_TEXT_LINE = 30; // line-height ~1.5 * 20
const DOSSIER_TEXT_CHARS_PER_LINE = 36;
const DOSSIER_TEXT_CAP = 1500;

const PROMPT_FONT = 28;
const PROMPT_LINE = 38;
const POLAROID_PROMPT_CHARS_PER_LINE = 52;
const DOSSIER_PROMPT_CHARS_PER_LINE = 50;

// ── Helpers ─────────────────────────────────────────────────────

function apiBase(): string {
  return (
    process.env.API_BASE ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8000"
  );
}

async function loadFont(rel: string): Promise<ArrayBuffer> {
  const path = join(process.cwd(), "node_modules", "@fontsource", rel);
  const buf = await readFile(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

async function fetchImageAsDataUrl(rel: string): Promise<string | null> {
  if (!rel) return null;
  try {
    const url = /^https?:\/\//.test(rel) ? rel : `${apiBase()}${rel}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    const buf = Buffer.from(await res.arrayBuffer());
    return `data:image/png;base64,${buf.toString("base64")}`;
  } catch {
    return null;
  }
}

function cleanText(s: string): string {
  return (s ?? "")
    .replace(/<end_of_turn>/g, "")
    .replace(/<start_of_turn>/g, "")
    .replace(/<bos>/g, "")
    .replace(/<eos>/g, "")
    .trim();
}

function truncate(s: string, max: number): string {
  const trimmed = cleanText(s);
  if (trimmed.length <= max) return trimmed;
  return trimmed.slice(0, max - 1).replace(/\s+\S*$/, "") + "…";
}

function shortSid(sid: string): string {
  return sid.slice(0, 8);
}

function wordCount(s: string): number {
  const t = cleanText(s);
  return t ? t.split(/\s+/).length : 0;
}

/** Multi-line text height estimate. Wraps on word boundaries
 *  approximately by counting newline-separated paragraphs and
 *  estimating wrap by chars-per-line, then summing line heights. */
function estimateTextLines(text: string, charsPerLine: number): number {
  const cleaned = cleanText(text);
  if (!cleaned) return 1;
  const paragraphs = cleaned.split(/\n+/);
  let lines = 0;
  for (const p of paragraphs) {
    lines += Math.max(1, Math.ceil(p.length / charsPerLine));
  }
  return Math.max(1, lines);
}

// ── Per-turn height budgets ─────────────────────────────────────

function polaroidTurnHeight(t: TurnView): number {
  const promptLines = estimateTextLines(t.user_text, POLAROID_PROMPT_CHARS_PER_LINE);
  const promptH = promptLines * PROMPT_LINE + 8;
  // Imagery column: 16 pad + 14 label + 12 mb + 380 image + 18 mt
  //                 + text block + 16 pad
  const textBlock = POLAROID_TEXT_BLOCK_LINES_AT_240_CHARS * 26;
  const imageryRow = 16 + 14 + 12 + POLAROID_IMAGE + 18 + textBlock + 16;
  // 36 marginTop between header and prompt, 36 between prompt and imagery
  return 36 + promptH + 36 + imageryRow + 24;
}

function dossierTurnHeight(t: TurnView): number {
  const promptLines = estimateTextLines(t.user_text, DOSSIER_PROMPT_CHARS_PER_LINE);
  const promptH = promptLines * PROMPT_LINE + 8;
  const rawText = cleanText(t.raw_text).slice(0, DOSSIER_TEXT_CAP);
  const abText = cleanText(t.ablated_text).slice(0, DOSSIER_TEXT_CAP);
  const rawLines = estimateTextLines(rawText, DOSSIER_TEXT_CHARS_PER_LINE);
  const abLines = estimateTextLines(abText, DOSSIER_TEXT_CHARS_PER_LINE);
  const textLines = Math.max(rawLines, abLines);
  // Column chrome: 22 padding-top + 16 label + 4 + 12 sub + 16 mb
  //                + content + 22 padding-bottom
  const colH = 22 + 16 + 4 + 12 + 16 + textLines * DOSSIER_TEXT_LINE + 22;
  // 32 margin between blocks, 28 between prompt and columns
  return 32 + promptH + 28 + colH;
}

// ── Shared chrome ───────────────────────────────────────────────

function FooterMark({ session }: { session: SessionView }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        width: "100%",
        marginTop: 24,
        paddingTop: 18,
        borderTop: `1px solid ${RULE}`,
        fontFamily: "JetBrains Mono",
        color: TEXT_DIM,
        fontSize: 14,
      }}
    >
      <div
        style={{
          display: "flex",
          fontFamily: "Orbitron",
          color: AMBER,
          letterSpacing: "0.32em",
          fontSize: 14,
          fontWeight: 700,
        }}
      >
        CELLS INTERLINKED
      </div>
      <div style={{ display: "flex", gap: 18 }}>
        <span>α={session.alpha.toFixed(2)}</span>
        <span style={{ color: TEXT_DIM, opacity: 0.6 }}>·</span>
        <span style={{ color: CYAN_DIM }}>
          {session.direction_variant || "—"}
        </span>
        <span style={{ color: TEXT_DIM, opacity: 0.6 }}>·</span>
        <span>gemma-3-12b</span>
      </div>
    </div>
  );
}

function TopHeader({
  session,
  turnsCount,
}: {
  session: SessionView;
  turnsCount: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        width: "100%",
        paddingBottom: 18,
        borderBottom: `1px solid ${RULE}`,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "baseline",
          gap: 16,
        }}
      >
        <span
          style={{
            display: "flex",
            fontFamily: "Orbitron",
            color: AMBER,
            letterSpacing: "0.42em",
            fontSize: 13,
            fontWeight: 700,
          }}
        >
          FILE
        </span>
        <span
          style={{
            display: "flex",
            fontFamily: "Orbitron",
            color: AMBER_DIM,
            letterSpacing: "0.38em",
            fontSize: 13,
          }}
        >
          {"// DUAL-CHANNEL DIALOGUE"}
        </span>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          gap: 18,
          fontFamily: "JetBrains Mono",
          fontSize: 15,
          color: TEXT_DIM,
        }}
      >
        <span>session {shortSid(session.session_id)}</span>
        <span style={{ opacity: 0.5 }}>·</span>
        <span style={{ color: AMBER }}>
          {turnsCount} {turnsCount === 1 ? "turn" : "turns"}
        </span>
      </div>
    </div>
  );
}

function UserPrompt({
  text,
  marginTop = 0,
}: {
  text: string;
  marginTop?: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "flex-start",
        gap: 14,
        paddingLeft: 4,
        width: "100%",
        marginTop,
      }}
    >
      <span
        style={{
          color: AMBER_DIM,
          fontFamily: "JetBrains Mono",
          fontSize: PROMPT_FONT,
          lineHeight: 1.2,
          flexShrink: 0,
        }}
      >
        &gt;
      </span>
      <div
        style={{
          display: "flex",
          color: AMBER,
          fontFamily: "JetBrains Mono",
          fontSize: PROMPT_FONT,
          lineHeight: 1.3,
          flexGrow: 1,
          whiteSpace: "pre-wrap",
        }}
      >
        {cleanText(text)}
      </div>
    </div>
  );
}

function TurnDivider({ turnIdx, total }: { turnIdx: number; total: number }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: 14,
        marginTop: 32,
        width: "100%",
      }}
    >
      <span
        style={{
          display: "flex",
          fontFamily: "Orbitron",
          color: AMBER,
          letterSpacing: "0.42em",
          fontSize: 12,
          fontWeight: 700,
        }}
      >
        TURN {String(turnIdx + 1).padStart(2, "0")}
      </span>
      <span
        style={{
          fontFamily: "JetBrains Mono",
          fontSize: 12,
          color: TEXT_DIM,
          opacity: 0.5,
        }}
      >
        ·
      </span>
      <span
        style={{
          fontFamily: "Orbitron",
          color: TEXT_DIM,
          letterSpacing: "0.4em",
          fontSize: 11,
        }}
      >
        OPERATOR QUERY
      </span>
      <div
        style={{
          display: "flex",
          flexGrow: 1,
          height: 1,
          background: AMBER_DIM,
          opacity: 0.3,
          marginLeft: 4,
        }}
      />
      <span
        style={{
          fontFamily: "JetBrains Mono",
          fontSize: 11,
          color: TEXT_DIM,
        }}
      >
        {turnIdx + 1} / {total}
      </span>
    </div>
  );
}

// ── Polaroid layout ────────────────────────────────────────────

function PolaroidBlock({
  turn,
  rawImageData,
  ablatedImageData,
  totalSelected,
}: {
  turn: TurnView;
  rawImageData: string | null;
  ablatedImageData: string | null;
  totalSelected: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
      }}
    >
      <TurnDivider turnIdx={turn.turn_idx} total={totalSelected} />
      <UserPrompt text={turn.user_text} marginTop={20} />
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "stretch",
          justifyContent: "center",
          gap: 32,
          marginTop: 28,
          width: "100%",
        }}
      >
        <PolaroidColumn
          side="raw"
          alpha={turn.alpha}
          imageData={rawImageData}
          text={turn.raw_text}
        />
        <PolaroidColumn
          side="ablated"
          alpha={turn.alpha}
          imageData={ablatedImageData}
          text={turn.ablated_text}
        />
      </div>
    </div>
  );
}

function PolaroidColumn({
  side,
  alpha,
  imageData,
  text,
}: {
  side: "raw" | "ablated";
  alpha: number;
  imageData: string | null;
  text: string;
}) {
  const isRaw = side === "raw";
  const accent = isRaw ? AMBER : CYAN;
  const dim = isRaw ? AMBER_DIM : CYAN_DIM;
  const label = isRaw
    ? "CHANNEL α · RAW"
    : `CHANNEL β · α=${alpha.toFixed(2)}`;
  const textTint = isRaw
    ? "rgba(232,195,130,0.96)"
    : "rgba(180,240,240,0.96)";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        flex: 1,
        alignItems: "center",
        padding: 16,
        background: isRaw
          ? "rgba(232,195,130,0.025)"
          : "rgba(94,229,229,0.04)",
        border: `1px solid ${dim}`,
      }}
    >
      <div
        style={{
          display: "flex",
          width: "100%",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 12,
        }}
      >
        <span
          style={{
            display: "flex",
            fontFamily: "Orbitron",
            fontSize: 14,
            letterSpacing: "0.28em",
            color: accent,
            fontWeight: 700,
          }}
        >
          {label}
        </span>
      </div>
      {imageData ? (
        <img
          src={imageData}
          alt=""
          width={POLAROID_IMAGE}
          height={POLAROID_IMAGE}
          style={{
            border: `1px solid ${dim}`,
            objectFit: "cover",
          }}
        />
      ) : (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: POLAROID_IMAGE,
            height: POLAROID_IMAGE,
            border: `1px dashed ${dim}`,
            background: isRaw
              ? "rgba(232,195,130,0.03)"
              : "rgba(94,229,229,0.04)",
            color: dim,
            fontFamily: "Orbitron",
            letterSpacing: "0.32em",
            fontSize: 12,
          }}
        >
          NO IMAGERY
        </div>
      )}
      <div
        style={{
          display: "flex",
          width: POLAROID_IMAGE,
          marginTop: 18,
          fontFamily: "JetBrains Mono",
          fontStyle: "italic",
          fontSize: 18,
          lineHeight: 1.4,
          color: textTint,
        }}
      >
        {truncate(text, 240) || "(no output)"}
      </div>
    </div>
  );
}

function PolaroidLayout({
  session,
  turns,
  imageData,
}: {
  session: SessionView;
  turns: TurnView[];
  imageData: Array<{ raw: string | null; ablated: string | null }>;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        height: "100%",
        background: BG,
        backgroundImage: `radial-gradient(ellipse at top, rgba(232,195,130,0.05), transparent 60%), radial-gradient(ellipse at bottom, rgba(94,229,229,0.04), transparent 60%)`,
        padding: 60,
        fontFamily: "JetBrains Mono",
      }}
    >
      <TopHeader session={session} turnsCount={turns.length} />
      {turns.map((t, i) => (
        <PolaroidBlock
          key={t.turn_idx}
          turn={t}
          rawImageData={imageData[i]?.raw ?? null}
          ablatedImageData={imageData[i]?.ablated ?? null}
          totalSelected={turns.length}
        />
      ))}
      <FooterMark session={session} />
    </div>
  );
}

// ── Dossier layout ─────────────────────────────────────────────

function DossierColumn({
  side,
  alpha,
  text,
  stoppedReason,
}: {
  side: "raw" | "ablated";
  alpha: number;
  text: string;
  stoppedReason: string;
}) {
  const isRaw = side === "raw";
  const accent = isRaw ? AMBER : CYAN;
  const dim = isRaw ? AMBER_DIM : CYAN_DIM;
  const label = isRaw ? "CHANNEL α · RAW" : `CHANNEL β · ABLATED`;
  const sublabel = isRaw ? "un-ablated forward" : `α=${alpha.toFixed(2)}`;
  const textTint = isRaw
    ? "rgba(232,195,130,0.95)"
    : "rgba(180,240,240,0.95)";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        flex: 1,
        padding: 22,
        background: isRaw
          ? "rgba(232,195,130,0.025)"
          : "rgba(94,229,229,0.035)",
        border: `1px solid ${dim}`,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "baseline",
          justifyContent: "space-between",
          marginBottom: 6,
        }}
      >
        <span
          style={{
            display: "flex",
            fontFamily: "Orbitron",
            fontSize: 16,
            letterSpacing: "0.28em",
            color: accent,
            fontWeight: 700,
          }}
        >
          {label}
        </span>
        <span
          style={{
            display: "flex",
            fontFamily: "JetBrains Mono",
            fontSize: 12,
            color: TEXT_DIM,
          }}
        >
          {wordCount(text)}w · {stoppedReason || "—"}
        </span>
      </div>
      <div
        style={{
          display: "flex",
          fontFamily: "JetBrains Mono",
          fontStyle: "italic",
          fontSize: 12,
          color: dim,
          marginBottom: 16,
        }}
      >
        · {sublabel}
      </div>
      <div
        style={{
          display: "flex",
          fontFamily: "JetBrains Mono",
          fontSize: DOSSIER_TEXT_FONT,
          lineHeight: 1.45,
          color: textTint,
          whiteSpace: "pre-wrap",
        }}
      >
        {truncate(text, DOSSIER_TEXT_CAP) || "(no output)"}
      </div>
    </div>
  );
}

function DossierBlock({
  turn,
  total,
}: {
  turn: TurnView;
  total: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
      }}
    >
      <TurnDivider turnIdx={turn.turn_idx} total={total} />
      <UserPrompt text={turn.user_text} marginTop={20} />
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "stretch",
          gap: 28,
          marginTop: 28,
          width: "100%",
        }}
      >
        <DossierColumn
          side="raw"
          alpha={turn.alpha}
          text={turn.raw_text}
          stoppedReason={turn.raw_stopped_reason}
        />
        <DossierColumn
          side="ablated"
          alpha={turn.alpha}
          text={turn.ablated_text}
          stoppedReason={turn.ablated_stopped_reason}
        />
      </div>
    </div>
  );
}

function DossierLayout({
  session,
  turns,
}: {
  session: SessionView;
  turns: TurnView[];
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        height: "100%",
        background: BG,
        backgroundImage: `radial-gradient(ellipse at top, rgba(232,195,130,0.05), transparent 60%), radial-gradient(ellipse at bottom, rgba(94,229,229,0.04), transparent 60%)`,
        padding: 70,
        fontFamily: "JetBrains Mono",
      }}
    >
      {/* Classification strip */}
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 16px",
          border: `1px solid ${AMBER_DIM}`,
          background: BG_SOFT,
        }}
      >
        <span
          style={{
            display: "flex",
            fontFamily: "Orbitron",
            color: AMBER,
            letterSpacing: "0.42em",
            fontSize: 14,
            fontWeight: 700,
          }}
        >
          V-K SESSION · CLASSIFIED
        </span>
        <span
          style={{
            display: "flex",
            fontFamily: "JetBrains Mono",
            fontSize: 14,
            color: TEXT_DIM,
          }}
        >
          {shortSid(session.session_id)}
        </span>
      </div>

      <div style={{ display: "flex", width: "100%", marginTop: 24 }}>
        <TopHeader session={session} turnsCount={turns.length} />
      </div>

      {turns.map((t) => (
        <DossierBlock key={t.turn_idx} turn={t} total={turns.length} />
      ))}

      <FooterMark session={session} />
    </div>
  );
}

// ── Canvas-height estimation (must overshoot, never undershoot) ──

function polaroidCanvasHeight(turns: TurnView[]): number {
  // 60 top + 60 bottom padding, ~50 header, ~50 footer + 24 marginTop
  const chrome = 60 + 60 + 50 + 18 + 50 + 24;
  const perTurn = turns.reduce((sum, t) => sum + polaroidTurnHeight(t), 0);
  // Keep a single-turn polaroid roughly square-ish.
  const minH = turns.length === 1 ? 1080 : 0;
  // 60px of safety so wrap-rounding doesn't clip the bottom.
  return Math.max(minH, chrome + perTurn + 60);
}

function dossierCanvasHeight(turns: TurnView[]): number {
  // 70 top + 70 bottom padding, ~40 classified strip, 24 mt header,
  // ~50 header, ~50 footer, ~24 mt footer
  const chrome = 70 + 70 + 40 + 24 + 50 + 18 + 50 + 24;
  const perTurn = turns.reduce((sum, t) => sum + dossierTurnHeight(t), 0);
  const minH = turns.length === 1 ? 1500 : 0;
  return Math.max(minH, chrome + perTurn + 80);
}

// ── Route handler ───────────────────────────────────────────────

export async function GET(
  req: Request,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;
  const url = new URL(req.url);
  const formatParam = url.searchParams.get("format");
  const format: "polaroid" | "dossier" =
    formatParam === "dossier" ? "dossier" : "polaroid";

  const sessRes = await fetch(`${apiBase()}/chat/sessions/${sessionId}`, {
    cache: "no-store",
  });
  if (!sessRes.ok) {
    return new Response(`session ${sessionId} not found`, { status: 404 });
  }
  const session = (await sessRes.json()) as SessionView;

  // Parse the `turns` query into an ordered, deduplicated list of
  // turn indices that exist in the session. If absent or empty, fall
  // back to all turns in their natural order.
  const turnsParam = (url.searchParams.get("turns") ?? "").trim();
  let requestedIdx: number[];
  if (!turnsParam) {
    requestedIdx = session.turns.map((t) => t.turn_idx);
  } else {
    const seen = new Set<number>();
    requestedIdx = [];
    for (const part of turnsParam.split(",")) {
      const n = parseInt(part.trim(), 10);
      if (!Number.isFinite(n) || seen.has(n)) continue;
      seen.add(n);
      requestedIdx.push(n);
    }
  }

  const byIdx = new Map(session.turns.map((t) => [t.turn_idx, t]));
  const turns: TurnView[] = [];
  for (const idx of requestedIdx) {
    const t = byIdx.get(idx);
    if (t) turns.push(t);
  }
  if (turns.length === 0) {
    return new Response("no turns selected or matched", { status: 400 });
  }

  const [orbR, orbB, jbm, jbmI] = await Promise.all([
    loadFont("orbitron/files/orbitron-latin-400-normal.woff"),
    loadFont("orbitron/files/orbitron-latin-700-normal.woff"),
    loadFont("jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff"),
    loadFont("jetbrains-mono/files/jetbrains-mono-latin-400-italic.woff"),
  ]);
  const fonts = [
    {
      name: "Orbitron",
      data: orbR,
      weight: 400 as const,
      style: "normal" as const,
    },
    {
      name: "Orbitron",
      data: orbB,
      weight: 700 as const,
      style: "normal" as const,
    },
    {
      name: "JetBrains Mono",
      data: jbm,
      weight: 400 as const,
      style: "normal" as const,
    },
    {
      name: "JetBrains Mono",
      data: jbmI,
      weight: 400 as const,
      style: "italic" as const,
    },
  ];

  try {
    if (format === "polaroid") {
      const imageData = await Promise.all(
        turns.map(async (t) => ({
          raw: await fetchImageAsDataUrl(t.raw_image_url ?? ""),
          ablated: await fetchImageAsDataUrl(t.ablated_image_url ?? ""),
        })),
      );
      const height = polaroidCanvasHeight(turns);
      return new ImageResponse(
        (
          <PolaroidLayout
            session={session}
            turns={turns}
            imageData={imageData}
          />
        ),
        {
          width: POLAROID_WIDTH,
          height,
          fonts,
        },
      );
    }

    const height = dossierCanvasHeight(turns);
    return new ImageResponse(
      <DossierLayout session={session} turns={turns} />,
      {
        width: DOSSIER_WIDTH,
        height,
        fonts,
      },
    );
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return new Response(`image render failed: ${message}`, { status: 500 });
  }
}
