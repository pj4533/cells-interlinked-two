export default function CaveatsPanel() {
  return (
    <div className="border border-warning/40 bg-bg-soft p-5 text-xs">
      <div className="font-display text-[10px] text-warning tracking-widest mb-3">
        read the fine print
      </div>
      <ul className="space-y-2 text-text-dim leading-relaxed">
        <li>
          <span className="text-text">This is not a consciousness test.</span> It is a
          coherence test between stated stance and computed state.
        </li>
        <li>
          The activation reading is performed by an{" "}
          <span className="font-mono text-amber-dim">NLA verbalizer (AV)</span> — Anthropic&apos;s
          May 2026 instrument. NLA outputs are <span className="text-text">constantly hypothetical</span>:
          the verbalizer can confabulate. Recurring claims across positions are more reliable than
          one-off ones.
        </li>
        <li>
          The Zhuokai/Li critique loaded this project by design: an NLA may pattern-match input
          surface features rather than read internal state. Until matched neutral controls are
          run, every divergence is suggestive, not load-bearing.
        </li>
        <li>
          The instrument reads at <span className="font-mono text-amber-dim">one trained layer</span>.
          What lives at other depths can be invisible to it. Where you read shapes what you see.
        </li>
        <li>
          Single-prompt results are noisy. Re-running the same probe samples a different point in
          the model&apos;s response distribution; consistency across runs is the meaningful signal.
        </li>
      </ul>
    </div>
  );
}
