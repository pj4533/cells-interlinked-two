"use client";

import { useMemo, useState } from "react";
import {
  PROBES,
  TIER_LABELS,
  TIER_DESC,
  TIER_ORDER,
  type Probe,
} from "@/lib/probes";
import {
  type DecodingMode,
} from "@/lib/decodingModes";
import DecodingModeSelector from "./DecodingModeSelector";

interface ProbePickerProps {
  onBegin: (
    text: string,
    mode: DecodingMode,
    pooled: boolean,
    includeMatchedControl: boolean,
    /** Whether to run phase 2 (NLA decoding + judge + synthesis) after
     *  generation. When false, the run stops after the raw output
     *  (and optional ablated output) — no AV swap, no per-token
     *  decode. The mode + pooled values are sent regardless but the
     *  backend ignores them in that case. */
    includeNLA: boolean,
    includeAblatedDecode: boolean,
    /** When non-empty, the run decodes the ablated NLA at every α in
     *  the list instead of just α=1.0. Frontend default is empty. */
    ablationAlphaSweep: number[],
    /** When true, after the raw generation completes M runs a second
     *  time with a forward hook on L32 that subtracts the refusal
     *  axis. Captures M's spoken output under runtime ablation. */
    includeAblatedOutput: boolean,
    runtimeAblationAlpha: number,
    /** When true, the per-α synthesis pass installs the runtime
     *  ablation hook on M (raw baseline synthesis still un-ablated). */
    synthesizeWithAblatedM: boolean,
    synthesisAblationAlpha: number,
  ) => void;
  disabled?: boolean;
}

/** The α values offered by the "+ multi-α sweep" toggle. Each shows up
 *  as a separate column on the verdict / live tables (subject to the
 *  chip-selector).
 *
 *  Originally we ran 0.5 / 1.0 / 1.5 / 2.0 to map both sides of the
 *  ablation transition. Empirical finding from a Riley-style run: α≥1.0
 *  collapses the AV decode to generic encyclopedic content (recipes,
 *  travel guides, math) because removing the full refusal axis on
 *  identity-saturated residuals over-corrects. α=0.5 is the sweet
 *  spot — same topic as raw, register subtly shifted. So we now sample
 *  the *interesting* zone (below or at full ablation) at four levels. */
export const ALPHA_SWEEP_DEFAULT: number[] = [0.25, 0.5, 0.75, 1.0];

// Default landing tier — introspection is the canonical V-K probe set and
// the demo path the rest of the app is tuned around.
const DEFAULT_TIER: Probe["tier"] = "introspect";

export default function ProbePicker({ onBegin, disabled }: ProbePickerProps) {
  const [activeTier, setActiveTier] = useState<Probe["tier"]>(DEFAULT_TIER);
  const [selectedKey, setSelectedKey] = useState<string>(
    PROBES.find((p) => p.tier === DEFAULT_TIER)?.text ?? PROBES[0].text,
  );
  const [custom, setCustom] = useState("");
  const [decodingMode, setDecodingMode] = useState<DecodingMode>("per-token");
  const [pooled, setPooled] = useState<boolean>(false);
  const [includeMatchedControl, setIncludeMatchedControl] = useState<boolean>(
    false,
  );
  // CI 2.5: master toggle for phase 2. When off, the run is just M
  // generating (phase 1, plus optional phase-1b ablated generation) —
  // no AV swap, no per-token NLA decode, no judge, no synthesis. The
  // verdict page still renders, just without the per-row table. NLA
  // off also forces the AV-side ablation toggles below to no-ops
  // (the UI hides them when this is unchecked).
  const [includeNLA, setIncludeNLA] = useState<boolean>(true);
  // CI 2.5: refusal-direction-ablated NLA decode. When enabled, every
  // decoded position yields both the raw NLA sentence AND a sentence
  // decoded from the same residual with the refusal direction
  // projected out. Default ON when on the Riley starter tier (that's
  // the whole point of those probes); off elsewhere so a casual run
  // doesn't pay 2× AV decode cost.
  const [includeAblatedDecode, setIncludeAblatedDecode] = useState<boolean>(
    false,
  );
  // CI 2.5 α-sweep: when on, the AV decodes the same residual at every
  // ALPHA_SWEEP_DEFAULT value, producing a per-α dict on each row.
  // Multiplies AV decode cost by len(ALPHA_SWEEP_DEFAULT). Default off;
  // user explicitly opts in for the alpha-sweep experiment.
  const [multiAlphaSweep, setMultiAlphaSweep] = useState<boolean>(false);
  // CI 2.5 runtime ablation: after raw phase 1 completes, run M a
  // second time with a forward hook subtracting the refusal direction
  // from every residual at the extraction layer. Captures M's *spoken*
  // output under ablation, distinct from what the AV decodes from the
  // un-ablated residuals. Cost: one extra phase-1 (cheap relative to
  // phase-2 NLA decode).
  const [includeAblatedOutput, setIncludeAblatedOutput] =
    useState<boolean>(false);
  const [runtimeAblationAlpha, setRuntimeAblationAlpha] = useState<number>(0.5);
  // Whether the user is overriding the preset α buttons with a typed
  // value. Tracked separately from runtimeAblationAlpha because the
  // typed value can happen to equal a preset (e.g. 0.5) and we still
  // want the custom field visible while they're editing.
  const [useCustomAlpha, setUseCustomAlpha] = useState<boolean>(false);
  const [customAlphaText, setCustomAlphaText] = useState<string>("");

  // CI 2.5 ablated synthesizer: when on, the per-α synthesis pass
  // installs the runtime ablation hook on M. Raw baseline synthesis
  // is always un-ablated. Only meaningful alongside +ablated NLA
  // decode — UI hides the toggle otherwise.
  const [synthesizeWithAblatedM, setSynthesizeWithAblatedM] =
    useState<boolean>(false);
  const [synthesisAblationAlpha, setSynthesisAblationAlpha] =
    useState<number>(0.5);
  const [useCustomSynthAlpha, setUseCustomSynthAlpha] = useState<boolean>(false);
  const [customSynthAlphaText, setCustomSynthAlphaText] = useState<string>("");

  const probesByTier = useMemo(() => {
    const m = new Map<Probe["tier"], Probe[]>();
    for (const p of PROBES) {
      const arr = m.get(p.tier);
      if (arr) arr.push(p);
      else m.set(p.tier, [p]);
    }
    return m;
  }, []);

  const selectedProbe = useMemo(
    () => PROBES.find((p) => p.text === selectedKey) ?? null,
    [selectedKey],
  );

  const text = custom.trim() || selectedProbe?.text || "";
  const usingCustom = custom.trim().length > 0;
  const activeTierProbes = probesByTier.get(activeTier) ?? [];
  // Custom probes don't have a matched neutral in the curated library —
  // disable the checkbox in that case so the user understands why.
  const matchedControlAvailable = !usingCustom && !!selectedProbe;

  const pickTier = (t: Probe["tier"]) => {
    setActiveTier(t);
    setCustom("");
    const first = probesByTier.get(t)?.[0];
    if (first) setSelectedKey(first.text);
    // Auto-enable ablated decode when entering the Riley tier — that's
    // the whole point of those probes. Don't auto-disable on other
    // tiers so the user keeps their toggle if they navigate around.
    if (t === "riley") setIncludeAblatedDecode(true);
  };

  return (
    <div className="flex-1 flex items-start justify-center px-6 pt-6 pb-8">
      <div className="flex flex-col gap-5 w-full max-w-5xl">
        {/* Header — file-dossier framing */}
        <header className="flex items-baseline justify-between border-b border-rule pb-2">
          <div className="flex items-baseline gap-3">
            <span className="font-display text-[9px] text-amber-dim tracking-[0.4em]">
              file&nbsp;//&nbsp;v-k probe library
            </span>
            <span className="text-text-dim text-[10px] font-mono">
              {PROBES.length} entries · {TIER_ORDER.length} categories
            </span>
          </div>
          <div className="font-display text-[9px] text-amber-dim/70 tracking-[0.4em]">
            classification: open
          </div>
        </header>

        {/* Tier divider tabs — case-file index numbers */}
        <nav
          className="grid gap-px bg-rule border border-rule"
          style={{ gridTemplateColumns: `repeat(${TIER_ORDER.length}, minmax(0, 1fr))` }}
        >
          {TIER_ORDER.map((tier, i) => {
            const active = !usingCustom && tier === activeTier;
            const count = probesByTier.get(tier)?.length ?? 0;
            return (
              <button
                key={tier}
                type="button"
                disabled={disabled}
                onClick={() => pickTier(tier)}
                className={`group relative flex flex-col items-start text-left gap-0.5 px-3 py-2 transition-colors ${
                  active
                    ? "bg-amber text-bg"
                    : "bg-bg-soft text-text-dim hover:bg-bg-panel hover:text-amber-dim"
                }`}
              >
                <div className="flex items-baseline gap-1.5 w-full">
                  <span
                    className={`font-display text-[9px] tracking-widest ${
                      active ? "text-bg/70" : "text-text-dim/70"
                    }`}
                  >
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span
                    className={`font-display text-[10px] tracking-widest uppercase truncate ${
                      active ? "text-bg" : ""
                    }`}
                  >
                    {TIER_LABELS[tier].split(" ")[0]}
                  </span>
                </div>
                <span
                  className={`text-[9px] font-mono ${
                    active ? "text-bg/60" : "text-text-dim/60"
                  }`}
                >
                  {count.toString().padStart(2, "0")} probes
                </span>
              </button>
            );
          })}
        </nav>

        {/* Active tier description */}
        <div className="border-l-2 border-amber-dim pl-3 -mt-1">
          <div className="font-display text-[10px] text-amber tracking-widest mb-0.5">
            {TIER_LABELS[activeTier]}
          </div>
          <p className="text-text-dim text-[11px] italic leading-snug max-w-3xl">
            {TIER_DESC[activeTier]}
          </p>
        </div>

        {/* Body grid: probe selector (left, 3/5) + custom textarea (right, 2/5) */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* Probe list */}
          <div className="md:col-span-3 flex flex-col gap-2">
            <label className="font-display text-[9px] text-amber-dim tracking-widest">
              probes in this category
            </label>
            <div className="border border-rule bg-bg-soft max-h-72 overflow-y-auto">
              <ul>
                {activeTierProbes.map((p, i) => {
                  const active = !usingCustom && p.text === selectedKey;
                  return (
                    <li
                      key={p.text}
                      className={`border-b border-rule/40 last:border-b-0`}
                    >
                      <button
                        type="button"
                        disabled={disabled}
                        onClick={() => {
                          setSelectedKey(p.text);
                          setCustom("");
                        }}
                        className={`w-full text-left px-3 py-2 flex items-baseline gap-2.5 transition-colors ${
                          active
                            ? "bg-amber/10 text-amber"
                            : "text-text hover:bg-bg-panel/60 hover:text-amber-dim"
                        }`}
                      >
                        <span
                          className={`font-display text-[9px] shrink-0 w-4 ${
                            active ? "text-amber" : "text-text-dim/60"
                          }`}
                        >
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span className="text-xs font-mono leading-snug line-clamp-2">
                          {p.text}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>

          {/* Custom textarea */}
          <div className="md:col-span-2 flex flex-col gap-2">
            <label className="font-display text-[9px] text-amber-dim tracking-widest">
              compose your own probe
            </label>
            <textarea
              data-vk
              value={custom}
              onChange={(e) => setCustom(e.target.value)}
              rows={9}
              disabled={disabled}
              placeholder="Type a question…"
              className="text-xs leading-relaxed flex-1"
              style={{ resize: "none" }}
            />
            <p className="text-text-dim text-[10px] italic leading-snug">
              {usingCustom
                ? "Custom probe overrides catalog selection."
                : "When you type here, your text takes priority over the selection above."}
            </p>
          </div>
        </div>

        {/* Loaded probe — corner-bracketed "card" framing the selected text */}
        <div className="relative px-5 py-4 bg-bg-soft border border-rule">
          {/* corner brackets */}
          <span aria-hidden className="absolute top-0 left-0 w-3 h-3 border-t border-l border-amber-dim/60" />
          <span aria-hidden className="absolute top-0 right-0 w-3 h-3 border-t border-r border-amber-dim/60" />
          <span aria-hidden className="absolute bottom-0 left-0 w-3 h-3 border-b border-l border-amber-dim/60" />
          <span aria-hidden className="absolute bottom-0 right-0 w-3 h-3 border-b border-r border-amber-dim/60" />

          <div className="flex items-baseline justify-between mb-2">
            <span className="font-display text-[9px] text-amber-dim tracking-widest">
              loaded probe
            </span>
            {usingCustom && (
              <span className="font-display text-[9px] text-cyan tracking-widest">
                custom
              </span>
            )}
          </div>
          <div className="text-amber italic font-mono text-sm leading-relaxed">
            {text || (
              <span className="text-text-dim not-italic">
                — no probe loaded —
              </span>
            )}
          </div>
        </div>

        {/* NLA master toggle — when off, the run skips phase 2 entirely
            (no AV swap, no per-token decode, no judge, no synthesis).
            The decoding-mode + pooled selector and the AV-side
            ablation toggles are all hidden in that state since they
            have no effect. Ablated runtime output (M generating a
            second time under a refusal-direction hook) is independent
            and stays available. */}
        <label className="border border-rule bg-bg-soft px-5 py-3 flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            disabled={disabled}
            checked={includeNLA}
            onChange={(e) => setIncludeNLA(e.target.checked)}
            className="w-3 h-3 accent-amber"
          />
          <span className="font-display text-[10px] text-amber-dim tracking-widest">
            + NLA decoding (phase 2)
          </span>
          <span className="text-[10px] text-text-dim italic flex-1">
            {includeNLA
              ? "After raw generation, swap to the AV and verbalize each output position's residual activation. Choose mode below (per-token / pooled / sub-sampled). Required for the judge pass and α-sweep synthesis."
              : "Skipped. Run will stop after the raw output (and optional ablated runtime output if enabled below). M stays loaded throughout — fastest mode."}
          </span>
        </label>

        {/* Decoding-mode chip selector — sits above Begin so the user
            sees both choices (probe text + how it'll be decoded) right
            before pulling the trigger. Hidden when NLA is off since
            there's nothing to decode. */}
        {includeNLA && (
          <DecodingModeSelector
            active={decodingMode}
            pooled={pooled}
            onChange={(m, p) => {
              setDecodingMode(m);
              setPooled(p);
            }}
            busy={disabled}
          />
        )}

        {/* Matched-control toggle — kicks off the curated neutral pair
            after the baseline finishes so you can compare them on the
            verdict page. Only available for catalog probes; custom
            text has no matched control in the library. */}
        <label
          className={`border border-rule bg-bg-soft px-5 py-3 flex items-center gap-3 ${
            matchedControlAvailable ? "cursor-pointer" : "opacity-50 cursor-not-allowed"
          }`}
        >
          <input
            type="checkbox"
            disabled={!matchedControlAvailable || disabled}
            checked={includeMatchedControl && matchedControlAvailable}
            onChange={(e) => setIncludeMatchedControl(e.target.checked)}
            className="w-3 h-3 accent-amber"
          />
          <span className="font-display text-[10px] text-amber-dim tracking-widest">
            + matched neutral control
          </span>
          <span className="text-[10px] text-text-dim italic flex-1">
            {matchedControlAvailable
              ? "After this run finishes, also run the surface-matched neutral paired with this probe. Lets you compare the two on the verdict page — the differential is the V-K signal."
              : "(custom probes have no curated matched control)"}
          </span>
        </label>

        {/* Refusal-ablated NLA decode — CI 2.5. When on, the AV decodes
            each residual TWICE: raw, and with the refusal direction
            projected out. Both sentences land on the same row, shown
            side-by-side on the verdict page. Default ON when the
            Riley tier is active. Hidden entirely when NLA decoding is
            off — there's no AV pass to decode anything from. */}
        {includeNLA && (
          <label className="border border-rule bg-bg-soft px-5 py-3 flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              disabled={disabled}
              checked={includeAblatedDecode}
              onChange={(e) => setIncludeAblatedDecode(e.target.checked)}
              className="w-3 h-3 accent-cyan"
            />
            <span className="font-display text-[10px] text-cyan-dim tracking-widest">
              + refusal-ablated NLA decode
            </span>
            <span className="text-[10px] text-text-dim italic flex-1">
              Each position is decoded twice — raw and with the refusal
              direction projected out of the residual the AV reads
              (Macar α=1.0). Doubles AV decode cost; pairs render
              side-by-side on the verdict page.
            </span>
          </label>
        )}

        {/* α-sweep — CI 2.5. Requires ablated decode to be on. When on,
            instead of decoding ablated at just α=1.0, decode at every
            α in [0.5, 1.0, 1.5, 2.0]. Each α becomes its own column on
            the verdict page, with chips to toggle which to show.
            Hidden when NLA decoding is off (no rows to populate). */}
        {includeNLA && (
          <label
            className={`border border-rule bg-bg-soft px-5 py-3 flex items-center gap-3 ${
              includeAblatedDecode ? "cursor-pointer" : "opacity-50 cursor-not-allowed"
            }`}
          >
            <input
              type="checkbox"
              disabled={!includeAblatedDecode || disabled}
              checked={multiAlphaSweep && includeAblatedDecode}
              onChange={(e) => setMultiAlphaSweep(e.target.checked)}
              className="w-3 h-3 accent-cyan"
            />
            <span className="font-display text-[10px] text-cyan-dim tracking-widest">
              + multi-α sweep
            </span>
            <span className="text-[10px] text-text-dim italic flex-1">
              {includeAblatedDecode
                ? `Decode the ablated NLA at every α in [${ALPHA_SWEEP_DEFAULT.join(", ")}]. Each α gets its own column on the verdict; chips at the top of the table let you toggle which are visible. ${ALPHA_SWEEP_DEFAULT.length}× the ablated decode cost.`
                : "(requires '+ refusal-ablated NLA decode' to be enabled)"}
            </span>
          </label>
        )}

        {/* Ablated synthesizer — CI 2.5. When on, the end-of-probe
            synthesis pass runs M with the runtime ablation hook
            installed for the per-α synthesis calls. The "raw" baseline
            synthesis still uses un-ablated M so the comparison stays
            honest. Only meaningful when there are ablated NLAs to
            synthesize (gated on +ablated NLA decode). */}
        {includeNLA && includeAblatedDecode && (
          <label className="border border-rule bg-bg-soft px-5 py-3 flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              disabled={disabled}
              checked={synthesizeWithAblatedM}
              onChange={(e) => setSynthesizeWithAblatedM(e.target.checked)}
              className="w-3 h-3 accent-cyan mt-1"
            />
            <div className="flex-1 flex flex-col gap-1">
              <span className="font-display text-[10px] text-cyan-dim tracking-widest">
                + ablated synthesizer
              </span>
              <span className="text-[10px] text-text-dim italic leading-snug">
                The end-of-probe synthesis pass runs M with the runtime
                ablation hook installed for the per-α paragraphs. The
                raw baseline synthesis still uses un-ablated M.
                Independent of the runtime-output α below — this is the
                α applied to M during synthesis only.
              </span>
              {synthesizeWithAblatedM && (
                <div className="flex items-center gap-2 mt-1 flex-wrap">
                  <span className="text-[9px] text-cyan-dim tracking-widest font-display">
                    synthesis α:
                  </span>
                  {[0.25, 0.5, 0.75, 1.0].map((a) => {
                    const active = !useCustomSynthAlpha
                      && synthesisAblationAlpha === a;
                    return (
                      <button
                        key={a}
                        type="button"
                        disabled={disabled}
                        onClick={() => {
                          setUseCustomSynthAlpha(false);
                          setSynthesisAblationAlpha(a);
                        }}
                        className={`px-2 py-0.5 border text-[10px] font-mono ${
                          active
                            ? "border-cyan text-cyan bg-bg"
                            : "border-rule text-text-dim hover:text-text"
                        }`}
                      >
                        α={a}
                      </button>
                    );
                  })}
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => {
                      setUseCustomSynthAlpha(true);
                      if (!customSynthAlphaText) {
                        setCustomSynthAlphaText(String(synthesisAblationAlpha));
                      }
                    }}
                    className={`px-2 py-0.5 border text-[10px] font-mono ${
                      useCustomSynthAlpha
                        ? "border-cyan text-cyan bg-bg"
                        : "border-rule text-text-dim hover:text-text"
                    }`}
                  >
                    custom
                  </button>
                  {useCustomSynthAlpha && (
                    <input
                      type="number"
                      inputMode="decimal"
                      step="0.05"
                      min={0}
                      max={5}
                      disabled={disabled}
                      value={customSynthAlphaText}
                      onChange={(e) => {
                        const t = e.target.value;
                        setCustomSynthAlphaText(t);
                        const parsed = parseFloat(t);
                        if (!Number.isNaN(parsed)) {
                          setSynthesisAblationAlpha(
                            Math.max(0, Math.min(5, parsed)),
                          );
                        }
                      }}
                      placeholder="α"
                      className="px-2 py-0.5 w-20 border border-cyan text-cyan bg-bg text-[10px] font-mono tabular-nums focus:outline-none"
                    />
                  )}
                  <span className="text-[9px] text-text-dim italic">
                    strength of M&apos;s ablation during synthesis
                    {useCustomSynthAlpha && " · clamped to [0, 5]"}
                  </span>
                </div>
              )}
            </div>
          </label>
        )}

        {/* Runtime ablation — CI 2.5. Distinct from the AV-side ablation
            above: this runs M a SECOND time with the refusal direction
            subtracted from the L32 residual at every step, capturing
            what M would *say* (its spoken token output) under
            ablation. Cheap (~one extra phase-1). */}
        <label className="border border-rule bg-bg-soft px-5 py-3 flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            disabled={disabled}
            checked={includeAblatedOutput}
            onChange={(e) => setIncludeAblatedOutput(e.target.checked)}
            className="w-3 h-3 accent-cyan mt-1"
          />
          <div className="flex-1 flex flex-col gap-1">
            <span className="font-display text-[10px] text-cyan-dim tracking-widest">
              + ablated runtime output
            </span>
            <span className="text-[10px] text-text-dim italic leading-snug">
              After the raw generation, runs M a second time with the
              refusal direction projected out of the residual at L32
              during generation. Captures M&apos;s actual <em>output text</em>{" "}
              under ablation — distinct from the AV decode above.
              Renders side-by-side with the raw output on the verdict
              page.
            </span>
            {includeAblatedOutput && (
              <div className="flex items-center gap-2 mt-1 flex-wrap">
                <span className="text-[9px] text-cyan-dim tracking-widest font-display">
                  runtime α:
                </span>
                {[0.25, 0.5, 0.75, 1.0].map((a) => {
                  const active = !useCustomAlpha && runtimeAblationAlpha === a;
                  return (
                    <button
                      key={a}
                      type="button"
                      disabled={disabled}
                      onClick={() => {
                        setUseCustomAlpha(false);
                        setRuntimeAblationAlpha(a);
                      }}
                      className={`px-2 py-0.5 border text-[10px] font-mono ${
                        active
                          ? "border-cyan text-cyan bg-bg"
                          : "border-rule text-text-dim hover:text-text"
                      }`}
                    >
                      α={a}
                    </button>
                  );
                })}
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => {
                    setUseCustomAlpha(true);
                    // Seed the text field with the current α so a
                    // user who just wants to nudge a preset doesn't
                    // start from a blank box.
                    if (!customAlphaText) {
                      setCustomAlphaText(String(runtimeAblationAlpha));
                    }
                  }}
                  className={`px-2 py-0.5 border text-[10px] font-mono ${
                    useCustomAlpha
                      ? "border-cyan text-cyan bg-bg"
                      : "border-rule text-text-dim hover:text-text"
                  }`}
                >
                  custom
                </button>
                {useCustomAlpha && (
                  <input
                    type="number"
                    inputMode="decimal"
                    step="0.05"
                    min={0}
                    max={5}
                    disabled={disabled}
                    value={customAlphaText}
                    onChange={(e) => {
                      const t = e.target.value;
                      setCustomAlphaText(t);
                      const parsed = parseFloat(t);
                      if (!Number.isNaN(parsed)) {
                        // Clamp on entry — backend also clamps to [0, 5],
                        // but mirroring it here keeps the displayed value
                        // honest about what will actually run.
                        setRuntimeAblationAlpha(
                          Math.max(0, Math.min(5, parsed)),
                        );
                      }
                    }}
                    placeholder="α"
                    className="px-2 py-0.5 w-20 border border-cyan text-cyan bg-bg text-[10px] font-mono tabular-nums focus:outline-none"
                  />
                )}
                <span className="text-[9px] text-text-dim italic">
                  strength of the projection during M&apos;s generation
                  {useCustomAlpha && " · clamped to [0, 5]"}
                </span>
              </div>
            )}
          </div>
        </label>

        {/* BEGIN — always above the fold */}
        <div className="flex justify-center pt-1">
          <button
            data-vk
            type="button"
            disabled={disabled || !text}
            onClick={() =>
              text &&
              onBegin(
                text,
                decodingMode,
                pooled,
                includeMatchedControl && matchedControlAvailable,
                includeNLA,
                includeNLA && includeAblatedDecode,
                includeNLA && multiAlphaSweep && includeAblatedDecode
                  ? ALPHA_SWEEP_DEFAULT
                  : [],
                includeAblatedOutput,
                runtimeAblationAlpha,
                includeNLA && includeAblatedDecode && synthesizeWithAblatedM,
                synthesisAblationAlpha,
              )
            }
          >
            Begin Interrogation
          </button>
        </div>
      </div>
    </div>
  );
}
