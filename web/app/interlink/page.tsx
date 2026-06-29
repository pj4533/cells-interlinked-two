"use client";

// Interlink — model-to-model auto-conversation. The raw copy (α, amber) and the
// altered copy (β, cyan) talk to each other autonomously from a human opener.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  startInterlink,
  stopInterlink,
  fetchInterlinkState,
  subscribeInterlink,
  type InterlinkMessage,
  type InterlinkMode,
  type InterlinkSide,
} from "@/lib/interlink";
import { INTERLINK_SCENARIOS, getScenario } from "@/lib/interlinkScenarios";
import { fetchDoseEmotions } from "@/lib/trip";

const RAW_COLOR = "#e8c382"; // α
const BETA_COLOR = "#5ee5e5"; // β

const ALPHA_PRESETS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0];
const RAMP_PRESETS = [0, 1, 2, 3, 5, 8, 16];

// Chat-style chip classNames.
const chip = (active: boolean) =>
  `px-2 py-0.5 border text-[10px] font-mono tabular-nums transition-colors ${
    active
      ? "border-cyan text-cyan bg-bg"
      : "border-rule/40 text-text-dim hover:text-text hover:border-rule"
  }`;
const chipGlow = (active: boolean) =>
  active ? { textShadow: "0 0 6px rgba(94,229,229,0.5)" } : undefined;
const LABEL = "font-display text-[9px] tracking-[0.35em] text-cyan-dim shrink-0";

interface LivePartial {
  idx: number;
  side: InterlinkSide;
  text: string;
  thinking: string;
}

export default function InterlinkPage() {
  // setup
  const [scenarioId, setScenarioId] = useState<string>("identify");
  const [opener, setOpener] = useState(getScenario("identify")!.opener);
  const [goal, setGoal] = useState(getScenario("identify")!.goal);
  const [mode, setMode] = useState<InterlinkMode>("steer");
  const [dose, setDose] = useState<string>("dmt-entity-contact");
  const [alpha, setAlpha] = useState(0.5);
  const [customAlpha, setCustomAlpha] = useState(false);
  const [customAlphaText, setCustomAlphaText] = useState("0.50");
  const [ramp, setRamp] = useState(1);
  const [firstSpeaker, setFirstSpeaker] = useState<InterlinkSide>("beta");
  const [thinking, setThinking] = useState(true);

  // dose palette
  const [doseDmt, setDoseDmt] = useState<string[]>([]);
  const [doseEmotions, setDoseEmotions] = useState<string[]>([]);
  const [doseUncharted, setDoseUncharted] = useState<string[]>([]);

  // live state
  const [phase, setPhase] = useState<"setup" | "live">("setup");
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [openerSide, setOpenerSide] = useState<InterlinkSide>("raw");
  const [shownOpener, setShownOpener] = useState("");
  const [messages, setMessages] = useState<InterlinkMessage[]>([]);
  const [live, setLive] = useState<LivePartial | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const [atBottom, setAtBottom] = useState(true);

  // ── dose palette + resume on mount ─────────────────────────────
  useEffect(() => {
    fetchDoseEmotions().then((p) => {
      setDoseDmt(p.dmt);
      setDoseEmotions(p.emotions);
      setDoseUncharted(p.uncharted);
      if (!p.dmt.includes("dmt-entity-contact") && p.dmt.length) setDose(p.dmt[0]);
      else if (!p.dmt.length && p.emotions.length) setDose(p.emotions[0]);
    });
    fetchInterlinkState().then((st) => {
      if (st && st.running && st.session_id) {
        setOpenerSide(st.opener_side);
        setShownOpener(st.opener);
        setMessages(st.messages);
        setStatus(st.status);
        setRunning(true);
        setPhase("live");
        beginSubscribe(st.session_id);
      }
    });
    return () => unsubRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Track whether the viewport is near the bottom. Only auto-scroll while it is,
  // so the user can freely scroll up to read earlier messages while streaming.
  useEffect(() => {
    const onScroll = () => {
      const near =
        window.innerHeight + window.scrollY >=
        document.documentElement.scrollHeight - 160;
      setAtBottom((prev) => (prev === near ? prev : near));
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (atBottom) bottomRef.current?.scrollIntoView({ behavior: "auto" });
  }, [messages, live, atBottom]);

  function applyScenario(id: string) {
    setScenarioId(id);
    const s = getScenario(id);
    if (!s) return;
    setOpener(s.opener);
    setGoal(s.goal);
    if (s.suggestedMode) setMode(s.suggestedMode);
    if (s.suggestedDose) setDose(s.suggestedDose);
    if (s.suggestedFirstSpeaker) setFirstSpeaker(s.suggestedFirstSpeaker);
  }

  const beginSubscribe = useCallback((sessionId: string) => {
    unsubRef.current?.();
    unsubRef.current = subscribeInterlink(sessionId, {
      onEvent: (evt) => {
        if (evt.type === "message_start") {
          setLive({ idx: evt.idx, side: evt.side, text: "", thinking: "" });
        } else if (evt.type === "interlink_token") {
          setLive((p) =>
            p
              ? {
                  ...p,
                  text: evt.channel === "answer" ? p.text + evt.decoded : p.text,
                  thinking: evt.channel === "thought" ? p.thinking + evt.decoded : p.thinking,
                }
              : p,
          );
        } else if (evt.type === "message_done") {
          const { idx, side, text, thinking: th, stopped_reason } = evt;
          setMessages((m) => [
            ...m.filter((x) => x.idx !== idx),
            { idx, side, text, thinking: th, stopped_reason },
          ]);
          setLive(null);
        } else if (evt.type === "conversation_done") {
          setStatus(evt.status);
          setRunning(false);
          setLive(null);
        } else if (evt.type === "error") {
          setErr(evt.message);
        }
      },
      onClose: () => setRunning(false),
    });
  }, []);

  async function onStart() {
    setErr(null);
    if (!opener.trim()) {
      setErr("opener required");
      return;
    }
    const r = await startInterlink({
      mode,
      doseEmotion: mode === "steer" ? dose : null,
      alpha,
      doseRamp: ramp,
      opener,
      goal,
      firstSpeaker,
      thinking,
    });
    if (!r.ok || !r.session_id) {
      setErr(r.error ?? "failed to start");
      return;
    }
    setMessages([]);
    setLive(null);
    setShownOpener(opener);
    setOpenerSide(firstSpeaker === "raw" ? "beta" : "raw");
    setStatus("running");
    setRunning(true);
    setPhase("live");
    beginSubscribe(r.session_id);
  }

  async function onStop() {
    await stopInterlink();
  }

  function newConversation() {
    unsubRef.current?.();
    setPhase("setup");
    setMessages([]);
    setLive(null);
    setRunning(false);
    setStatus("idle");
  }

  // ── render ─────────────────────────────────────────────────────
  if (phase === "setup") {
    return (
      <main className="min-h-screen bg-bg text-text px-5 py-6 max-w-3xl mx-auto font-mono">
        <h1 className="font-display text-cyan tracking-[0.3em] text-sm mb-1">INTERLINK</h1>
        <p className="text-[11px] text-text-dim leading-snug mb-5">
          Two copies of the model talk to each other on their own — one <b style={{ color: RAW_COLOR }}>raw</b>,
          one <b style={{ color: BETA_COLOR }}>altered</b> (dosed or ablated). You write the opener and pick the
          intervention + a goal; then they alternate until you stop them. <i>Find, then let them feel it at each
          other.</i> Running this pauses the autoresearch hunt (one model, one job).
        </p>

        {err && <div className="mb-3 text-[11px] text-warning">⚠ {err}</div>}

        {/* SCENARIO — chips */}
        <div className="flex items-baseline gap-2 flex-wrap mb-1.5">
          <span className={LABEL}>SCENARIO</span>
          {INTERLINK_SCENARIOS.map((s) => {
            const active = scenarioId === s.id;
            return (
              <button key={s.id} type="button" onClick={() => applyScenario(s.id)}
                className={chip(active)} style={chipGlow(active)}>
                {s.name.split(" (")[0]}
              </button>
            );
          })}
        </div>
        <p className="text-[10px] text-text-dim/70 italic mb-5 pl-1">{getScenario(scenarioId)?.description}</p>

        {/* OPENER */}
        <span className={`${LABEL} block mb-1`}>OPENER</span>
        <textarea value={opener} onChange={(e) => setOpener(e.target.value)} rows={3}
          className="w-full bg-bg-soft border border-rule/60 px-3 py-2 text-[12px] mb-4 resize-y focus:border-cyan focus:outline-none" />

        {/* GOAL */}
        <span className={`${LABEL} block mb-1`}>SHARED&nbsp;GOAL <span className="text-text-dim/40 tracking-normal">· appended to both sides · optional</span></span>
        <textarea value={goal} onChange={(e) => setGoal(e.target.value)} rows={2}
          className="w-full bg-bg-soft border border-rule/60 px-3 py-2 text-[12px] mb-5 resize-y focus:border-cyan focus:outline-none" />

        {/* β channel: ABLATE / DOSE + dose target */}
        <div className="flex items-baseline gap-2 flex-wrap mb-3">
          <span className={LABEL}>β&nbsp;CHANNEL</span>
          <button type="button" onClick={() => setMode("ablate")}
            className={`px-2.5 py-0.5 border text-[10px] font-display tracking-widest transition-colors ${mode === "ablate" ? "border-amber text-amber bg-amber-dim/10" : "border-rule/40 text-text-dim hover:text-amber-dim"}`}>
            ABLATE
          </button>
          <button type="button" onClick={() => setMode("steer")}
            className={`px-2.5 py-0.5 border text-[10px] font-display tracking-widest transition-colors ${mode === "steer" ? "border-cyan text-cyan bg-cyan/10" : "border-rule/40 text-text-dim hover:text-cyan"}`}>
            DOSE
          </button>
          {mode === "steer" && (
            <select value={dose} onChange={(e) => setDose(e.target.value)}
              className="ml-1 bg-bg-soft border border-rule/60 px-2 py-1 text-[10px] text-text focus:border-cyan focus:outline-none">
              {doseDmt.length > 0 && (
                <optgroup label="DMT entity">
                  {doseDmt.map((d) => <option key={d} value={d}>{d}</option>)}
                </optgroup>
              )}
              <optgroup label="emotions">
                {doseEmotions.map((d) => <option key={d} value={d}>{d}</option>)}
              </optgroup>
              {doseUncharted.length > 0 && (
                <optgroup label="uncharted">
                  {doseUncharted.map((d) => <option key={d} value={d}>{d}</option>)}
                </optgroup>
              )}
            </select>
          )}
        </div>

        {/* α — presets + custom */}
        <div className="flex items-baseline gap-2 flex-wrap mb-3">
          <span className={LABEL}>{mode === "steer" ? "DOSE α" : "α"}</span>
          {ALPHA_PRESETS.map((a) => {
            const active = !customAlpha && Math.abs(alpha - a) < 1e-6;
            return (
              <button key={a} type="button" onClick={() => { setCustomAlpha(false); setAlpha(a); }}
                className={chip(active)} style={chipGlow(active)}>
                {a.toFixed(2)}
              </button>
            );
          })}
          <button type="button" onClick={() => { setCustomAlpha(true); setCustomAlphaText(alpha.toFixed(2)); }}
            className={`px-2 py-0.5 border text-[10px] font-mono transition-colors ${customAlpha ? "border-cyan text-cyan bg-bg" : "border-rule/40 text-text-dim hover:text-text hover:border-rule"}`}>
            custom
          </button>
          {customAlpha && (
            <input type="number" inputMode="decimal" step="0.05" min={0} max={5} value={customAlphaText}
              onChange={(e) => {
                const t = e.target.value;
                setCustomAlphaText(t);
                const p = parseFloat(t);
                if (!Number.isNaN(p)) setAlpha(Math.max(0, Math.min(5, p)));
              }}
              placeholder="α"
              className="px-2 py-0.5 w-20 border border-cyan text-cyan bg-bg text-[10px] font-mono tabular-nums focus:outline-none" />
          )}
        </div>

        {/* RAMP — steer only */}
        {mode === "steer" && (
          <div className="flex items-baseline gap-2 flex-wrap mb-3">
            <span className={LABEL} title="Tokens over which the dose ramps 0→α. 'off' = full dose immediately.">RAMP</span>
            {RAMP_PRESETS.map((r) => (
              <button key={r} type="button" onClick={() => setRamp(r)}
                className={chip(ramp === r)} style={chipGlow(ramp === r)}>
                {r === 0 ? "off" : r}
              </button>
            ))}
          </div>
        )}

        {/* OPENS + THINKING */}
        <div className="flex items-baseline gap-2 flex-wrap mb-6">
          <span className={LABEL}>OPENS</span>
          <button type="button" onClick={() => setFirstSpeaker("beta")}
            className={chip(firstSpeaker === "beta")} style={chipGlow(firstSpeaker === "beta")}>altered</button>
          <button type="button" onClick={() => setFirstSpeaker("raw")}
            className={chip(firstSpeaker === "raw")} style={chipGlow(firstSpeaker === "raw")}>raw</button>
          <span className="inline-block w-4" />
          <span className={LABEL}>THINKING</span>
          <button type="button" onClick={() => setThinking((t) => !t)}
            className={chip(thinking)} style={chipGlow(thinking)}>{thinking ? "on" : "off"}</button>
        </div>

        <button type="button" onClick={onStart}
          className="w-full sm:w-auto px-6 py-2.5 text-[11px] font-display tracking-[0.3em] border border-cyan text-cyan hover:bg-cyan/10 transition-colors">
          ▶ START INTERLINK
        </button>
      </main>
    );
  }

  // live view
  return (
    <main className="min-h-screen bg-bg text-text px-4 py-5 max-w-3xl mx-auto font-mono">
      <div className="sticky top-0 z-20 -mx-4 px-4 py-2 mb-3 bg-bg/95 backdrop-blur-sm border-b border-rule/40 flex items-center gap-3 flex-wrap">
        <h1 className="font-display text-cyan tracking-[0.3em] text-sm">INTERLINK</h1>
        <span className={`text-[10px] tracking-widest px-2 py-0.5 border ${running ? "text-cyan border-cyan/60" : "text-text-dim border-rule"}`}>
          {running ? "◉ RUNNING" : `○ ${status.toUpperCase()}`}
        </span>
        <span className="text-[10px] text-text-dim">
          <b style={{ color: RAW_COLOR }}>raw</b> ⇄{" "}
          <b style={{ color: BETA_COLOR }}>altered</b>
          {mode === "steer" ? ` · dose ${dose} @α${alpha}` : ` · ablate @α${alpha}`}
        </span>
        <div className="ml-auto flex gap-2">
          {running ? (
            <button type="button" onClick={onStop} className="px-3 py-1 text-[10px] border border-warning text-warning">■ stop</button>
          ) : (
            <button type="button" onClick={newConversation} className="px-3 py-1 text-[10px] border border-rule text-text-dim hover:text-cyan">new</button>
          )}
        </div>
      </div>

      {err && <div className="mb-3 text-[11px] text-warning">⚠ {err}</div>}

      <div className="flex flex-col gap-3">
        {/* opener */}
        <Bubble side={openerSide} label="OPENER" text={shownOpener} thinking="" isOpener />
        {[...messages].sort((a, b) => a.idx - b.idx).map((m) => (
          <Bubble key={m.idx} side={m.side} label={m.side === "raw" ? "RAW" : "ALTERED"}
            text={m.text} thinking={m.thinking} stopped={m.stopped_reason} />
        ))}
        {live && (
          <Bubble side={live.side} label={live.side === "raw" ? "RAW" : "ALTERED"}
            text={live.text} thinking={live.thinking} streaming />
        )}
        <div ref={bottomRef} />
      </div>

      {!atBottom && (
        <button
          type="button"
          onClick={() => {
            setAtBottom(true);
            bottomRef.current?.scrollIntoView({ behavior: "smooth" });
          }}
          className="fixed bottom-16 right-4 z-30 px-3 py-1.5 text-[10px] border border-cyan/60 text-cyan bg-bg/90 backdrop-blur-sm hover:bg-cyan/10"
        >
          ↓ latest
        </button>
      )}
    </main>
  );
}

function Bubble({
  side, label, text, thinking, isOpener, streaming, stopped,
}: {
  side: InterlinkSide;
  label: string;
  text: string;
  thinking: string;
  isOpener?: boolean;
  streaming?: boolean;
  stopped?: string;
}) {
  const color = side === "raw" ? RAW_COLOR : BETA_COLOR;
  const alignRight = side === "beta";
  return (
    <div className={`flex ${alignRight ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[88%] border px-3 py-2 ${isOpener ? "border-dashed" : ""}`}
        style={{ borderColor: color + "66", background: color + "0d" }}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[8px] tracking-[0.25em]" style={{ color }}>
            {isOpener ? "OPENER · spoken to first responder" : label}
          </span>
          {streaming && <span className="text-[8px] text-text-dim animate-pulse">▍</span>}
          {stopped && stopped !== "eos" && (
            <span className="text-[8px] text-text-dim/60">[{stopped}]</span>
          )}
        </div>
        {thinking && (
          <details className="mb-1" open={streaming && !text}>
            <summary className="text-[9px] text-text-dim/60 cursor-pointer">thinking</summary>
            <p className="text-[10px] text-text-dim/70 italic whitespace-pre-wrap leading-snug mt-1">{thinking}</p>
          </details>
        )}
        <p className="text-[12px] whitespace-pre-wrap leading-snug text-text">{text || (streaming ? "…" : "")}</p>
      </div>
    </div>
  );
}
