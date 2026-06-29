/** Server-rendered share card for an Interlink (model-to-model) conversation.
 *
 *  GET /share/interlink/{sessionId}?msgs=0,1,2  → image/png
 *
 *  `msgs` is a comma-separated list of message idx values (order honoured); if
 *  absent, the whole conversation. Each message becomes a full-width block tinted
 *  by side (raw = amber, altered = cyan). The opener + session setup (β dose/α)
 *  are stamped at the top. No URL on the artifact — the image is the artifact.
 */

import { ImageResponse } from "next/og";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface Msg {
  idx: number;
  side: "raw" | "beta";
  text: string;
  stopped_reason: string;
}
interface Session {
  session_id: string;
  mode: string;
  dose_emotion: string | null;
  alpha: number;
  first_speaker: string;
  opener: string;
  goal: string;
  status: string;
  messages: Msg[];
}

const BG = "#0a0d10";
const AMBER = "#e8c382";
const AMBER_DIM = "#8a7349";
const CYAN = "#5ee5e5";
const CYAN_DIM = "#3a8d8d";
const TEXT = "#c6ccd2";
const TEXT_DIM = "#6f7a83";
const RULE = "#2a3038";

const WIDTH = 1080;
const TEXT_FONT = 20;
const TEXT_LINE = 30;
const CHARS_PER_LINE = 52; // intentionally low → overestimate lines → never clip
const TEXT_CAP = 1800;

function apiBase(): string {
  return process.env.API_BASE ?? process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
async function loadFont(rel: string): Promise<ArrayBuffer> {
  const path = join(process.cwd(), "node_modules", "@fontsource", rel);
  const buf = await readFile(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}
function cleanText(s: string): string {
  return (s ?? "").replace(/<end_of_turn>|<start_of_turn>|<bos>|<eos>|<turn\|>/g, "").trim();
}
function estimateLines(text: string, cpl: number): number {
  const c = cleanText(text);
  if (!c) return 1;
  let lines = 0;
  for (const p of c.split(/\n+/)) lines += Math.max(1, Math.ceil(p.length / cpl));
  return Math.max(1, lines);
}
function blockHeight(text: string): number {
  const lines = estimateLines(cleanText(text).slice(0, TEXT_CAP), CHARS_PER_LINE);
  // 16 pad-top + 18 label + 12 + content + 16 pad-bottom + 18 gap
  return 16 + 18 + 12 + lines * TEXT_LINE + 16 + 18;
}

function MsgBlock({ side, text, stopped }: { side: "raw" | "beta" | "opener"; text: string; stopped?: string }) {
  const isRaw = side === "raw";
  const isOpener = side === "opener";
  const accent = isOpener ? TEXT_DIM : isRaw ? AMBER : CYAN;
  const dim = isOpener ? RULE : isRaw ? AMBER_DIM : CYAN_DIM;
  const label = isOpener ? "OPENER" : isRaw ? "CHANNEL α · RAW" : "CHANNEL β · ALTERED";
  return (
    <div style={{
      display: "flex", flexDirection: "column", width: "100%", marginTop: 18,
      padding: 16, border: `1px ${isOpener ? "dashed" : "solid"} ${dim}`,
      background: isRaw ? "rgba(232,195,130,0.03)" : isOpener ? "rgba(255,255,255,0.015)" : "rgba(94,229,229,0.04)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <span style={{ display: "flex", fontFamily: "Orbitron", fontSize: 14, letterSpacing: "0.28em", color: accent, fontWeight: 700 }}>{label}</span>
        {stopped && stopped !== "eos" && (
          <span style={{ display: "flex", fontFamily: "JetBrains Mono", fontSize: 12, color: TEXT_DIM }}>[{stopped}]</span>
        )}
      </div>
      <div style={{ display: "flex", fontFamily: "JetBrains Mono", fontSize: TEXT_FONT, lineHeight: 1.45, color: TEXT, whiteSpace: "pre-wrap" }}>
        {cleanText(text).slice(0, TEXT_CAP) || "(no output)"}
      </div>
    </div>
  );
}

export async function GET(req: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  const url = new URL(req.url);

  const res = await fetch(`${apiBase()}/interlink/sessions/${sessionId}`, { cache: "no-store" });
  if (!res.ok) return new Response(`session ${sessionId} not found`, { status: 404 });
  const session = (await res.json()) as Session;

  const msgsParam = (url.searchParams.get("msgs") ?? "").trim();
  let idxs: number[];
  if (!msgsParam) {
    idxs = session.messages.map((m) => m.idx);
  } else {
    const seen = new Set<number>();
    idxs = [];
    for (const p of msgsParam.split(",")) {
      const n = parseInt(p.trim(), 10);
      if (!Number.isFinite(n) || seen.has(n)) continue;
      seen.add(n);
      idxs.push(n);
    }
  }
  const byIdx = new Map(session.messages.map((m) => [m.idx, m]));
  const msgs = idxs.map((i) => byIdx.get(i)).filter((m): m is Msg => !!m);
  if (msgs.length === 0) return new Response("no messages selected", { status: 400 });

  const [orbR, orbB, jbm, jbmI] = await Promise.all([
    loadFont("orbitron/files/orbitron-latin-400-normal.woff"),
    loadFont("orbitron/files/orbitron-latin-700-normal.woff"),
    loadFont("jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff"),
    loadFont("jetbrains-mono/files/jetbrains-mono-latin-400-italic.woff"),
  ]);
  const fonts = [
    { name: "Orbitron", data: orbR, weight: 400 as const, style: "normal" as const },
    { name: "Orbitron", data: orbB, weight: 700 as const, style: "normal" as const },
    { name: "JetBrains Mono", data: jbm, weight: 400 as const, style: "normal" as const },
    { name: "JetBrains Mono", data: jbmI, weight: 400 as const, style: "italic" as const },
  ];

  const setupLine =
    session.mode === "steer"
      ? `dose ${session.dose_emotion} · α=${session.alpha.toFixed(2)}`
      : `refusal ablation · α=${session.alpha.toFixed(2)}`;

  const contentH = msgs.reduce((s, m) => s + blockHeight(m.text), 0);
  const openerH = blockHeight(session.opener);
  const chrome = 70 + 70 + 50 + 18 + 50 + 24 + 80; // pad + header + footer + safety
  const height = Math.max(900, chrome + openerH + contentH);

  try {
    return new ImageResponse(
      (
        <div style={{
          display: "flex", flexDirection: "column", width: "100%", height: "100%", background: BG,
          backgroundImage: `radial-gradient(ellipse at top, rgba(94,229,229,0.05), transparent 60%), radial-gradient(ellipse at bottom, rgba(232,195,130,0.04), transparent 60%)`,
          padding: 70, fontFamily: "JetBrains Mono",
        }}>
          {/* header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%", paddingBottom: 18, borderBottom: `1px solid ${RULE}` }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
              <span style={{ display: "flex", fontFamily: "Orbitron", color: CYAN, letterSpacing: "0.42em", fontSize: 14, fontWeight: 700 }}>INTERLINK</span>
              <span style={{ display: "flex", fontFamily: "Orbitron", color: TEXT_DIM, letterSpacing: "0.32em", fontSize: 12 }}>{"// RAW <-> ALTERED"}</span>
            </div>
            <div style={{ display: "flex", gap: 16, fontFamily: "JetBrains Mono", fontSize: 14, color: TEXT_DIM }}>
              <span style={{ color: CYAN_DIM }}>{setupLine}</span>
            </div>
          </div>

          <MsgBlock side="opener" text={session.opener} />
          {msgs.map((m) => (
            <MsgBlock key={m.idx} side={m.side} text={m.text} stopped={m.stopped_reason} />
          ))}

          {/* footer */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%", marginTop: 24, paddingTop: 18, borderTop: `1px solid ${RULE}`, fontFamily: "JetBrains Mono", color: TEXT_DIM, fontSize: 14 }}>
            <span style={{ display: "flex", fontFamily: "Orbitron", color: AMBER, letterSpacing: "0.32em", fontSize: 14, fontWeight: 700 }}>CELLS INTERLINKED</span>
            <div style={{ display: "flex", gap: 16 }}>
              <span>{session.session_id.slice(0, 8)}</span>
              <span style={{ opacity: 0.6 }}>·</span>
              <span>gemma-4-12b</span>
            </div>
          </div>
        </div>
      ),
      { width: WIDTH, height, fonts },
    );
  } catch (e) {
    return new Response(`image render failed: ${e instanceof Error ? e.message : String(e)}`, { status: 500 });
  }
}
