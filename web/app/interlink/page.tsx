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

const ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
const RAMPS = [0, 1, 2, 4, 8, 16];

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

        <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">SCENARIO</label>
        <select
          value={scenarioId}
          onChange={(e) => applyScenario(e.target.value)}
          className="w-full bg-bg-soft border border-rule px-2 py-1.5 text-[11px] mb-1"
        >
          {INTERLINK_SCENARIOS.map((s) => (
            <option key={s.id} value={s.id}>{s.name}</option>
          ))}
        </select>
        <p className="text-[10px] text-text-dim/70 italic mb-4">{getScenario(scenarioId)?.description}</p>

        <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">OPENER (the kickoff message)</label>
        <textarea
          value={opener}
          onChange={(e) => setOpener(e.target.value)}
          rows={3}
          className="w-full bg-bg-soft border border-rule px-2 py-1.5 text-[11px] mb-4 resize-y"
        />

        <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">
          SHARED GOAL (appended to both sides' context — optional)
        </label>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={2}
          className="w-full bg-bg-soft border border-rule px-2 py-1.5 text-[11px] mb-4 resize-y"
        />

        <div className="flex flex-wrap gap-4 mb-4">
          {/* mode */}
          <div>
            <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">β INTERVENTION</label>
            <div className="flex gap-1">
              {(["steer", "ablate"] as InterlinkMode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={`px-3 py-1 text-[10px] border ${mode === m ? "border-cyan text-cyan" : "border-rule text-text-dim"}`}
                >
                  {m === "steer" ? "dose" : "ablate"}
                </button>
              ))}
            </div>
          </div>
          {/* dose target */}
          {mode === "steer" && (
            <div className="min-w-[14rem]">
              <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">DOSE</label>
              <select
                value={dose}
                onChange={(e) => setDose(e.target.value)}
                className="w-full bg-bg-soft border border-rule px-2 py-1 text-[10px]"
              >
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
            </div>
          )}
          {/* alpha */}
          <div>
            <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">α (strength)</label>
            <select value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="bg-bg-soft border border-rule px-2 py-1 text-[10px]">
              {ALPHAS.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
          {/* ramp */}
          {mode === "steer" && (
            <div>
              <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">RAMP</label>
              <select value={ramp} onChange={(e) => setRamp(parseInt(e.target.value, 10))}
                className="bg-bg-soft border border-rule px-2 py-1 text-[10px]">
                {RAMPS.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
          )}
          {/* first speaker */}
          <div>
            <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">OPENS</label>
            <div className="flex gap-1">
              {(["beta", "raw"] as InterlinkSide[]).map((s) => (
                <button key={s} type="button" onClick={() => setFirstSpeaker(s)}
                  className={`px-3 py-1 text-[10px] border ${firstSpeaker === s ? "border-cyan text-cyan" : "border-rule text-text-dim"}`}>
                  {s === "beta" ? "altered" : "raw"}
                </button>
              ))}
            </div>
          </div>
          {/* thinking */}
          <div>
            <label className="block text-[9px] tracking-widest text-text-dim/70 mb-1">THINKING</label>
            <button type="button" onClick={() => setThinking((t) => !t)}
              className={`px-3 py-1 text-[10px] border ${thinking ? "border-cyan text-cyan" : "border-rule text-text-dim"}`}>
              {thinking ? "on" : "off"}
            </button>
          </div>
        </div>

        <button type="button" onClick={onStart}
          className="px-5 py-2 text-[11px] border border-cyan text-cyan hover:bg-cyan/10 tracking-widest">
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
