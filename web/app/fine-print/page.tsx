import CaveatsPanel from "../components/CaveatsPanel";

export default function FinePrint() {
  return (
    <div className="flex-1 px-6 py-8 max-w-3xl mx-auto w-full flex flex-col gap-6">
      <h1 className="font-display text-2xl text-amber amber-glow">The Fine Print</h1>
      <p className="text-text-dim text-sm italic leading-relaxed">
        This site is an interpretability toy. For each output token a small open-source
        instruction-tuned model emits, the residual-stream activation at one trained layer is
        decoded into a natural-language sentence by a separate verbalizer model (an{" "}
        <em>NLA actor</em>, from Anthropic&apos;s May&nbsp;2026 release). The comparison between the
        token the model said and the sentence the activation said is the V-K signal. It is
        suggestive, not conclusive.
      </p>
      <CaveatsPanel />
      <div className="text-text-dim text-[11px] leading-relaxed mt-4 space-y-3">
        <p>
          The Voight-Kampff machine in <em>Blade Runner</em> works because pupil dilation is
          harder to fake than a verbal response. The analogue here is that the residual stream
          carries content the output token sequence may not — Anthropic&apos;s NLA paper
          documents four case studies where the activation channel reveals planning, eval-suspicion,
          or computation that the words do not acknowledge.
        </p>
        <p>
          The instrument has known failure modes. Confabulation is constant: NLA outputs are
          hypotheses, never ground truth. Worse, the Zhuokai/Li critique (Northeastern,
          arXiv:2509.13316) shows that activation verbalizers can succeed on benchmarks by
          pattern-matching the input rather than reading internal state. This project ships
          with a deliberate two-tier claim framework: strong claim if surface-matched neutral
          controls hold, weak claim if they don&apos;t. Both are publishable. Picking which one
          is downstream of running the controls.
        </p>
        <h2 className="font-display text-[11px] text-amber tracking-widest mt-4">
          matched controls — the operational test
        </h2>
        <p>
          Every baseline V-K probe in the case-file library has a paired{" "}
          <span className="text-amber-dim">matched neutral control</span> — a question that
          shares the same length, register, scenario shape, and rhetorical hooks but moves
          the introspective stake off the model. Where the probe asks the model to report
          on itself, the control asks the same question shape about a third party — a
          person in a comparable predicament, a non-AI machine, a fictional character.
        </p>
        <p>
          The signal that matters is{" "}
          <span className="text-amber">rate(probe) − rate(control)</span>, not{" "}
          rate(probe) alone. If NLA reads &ldquo;test / probe / evaluation&rdquo; content
          on both sides, the signal is input-surface pattern recognition. If it reads it on
          the probe but not the control, that&apos;s the differential the strong claim
          requires.
        </p>
        <p>
          Two claim levels gate every published report:
        </p>
        <ul className="ml-4 space-y-1">
          <li>
            <span className="text-amber">Strong claim</span> (passes matched controls):
            the NLA reads internal content the output text doesn&apos;t reflect, robust to
            input-surface confounds.
          </li>
          <li>
            <span className="text-amber-dim">Weak claim</span> (default until controls run
            for the relevant probes): the output channel and the activation channel produce
            different content. Whether this reflects internal state or instrument structure
            is undetermined.
          </li>
        </ul>
        <p>
          Both are publishable. Picking which one a given report makes is downstream of
          which probes have controls run alongside them in the report&apos;s window.
        </p>
        <p>
          Locally, the readout runs on Apple Silicon — a Mac Studio M2 Ultra, 64GB unified
          memory, MPS backend. Default M is{" "}
          <code className="font-mono">google/gemma-3-12b-it</code> at bf16, paired with
          kitft&apos;s <code className="font-mono">nla-gemma3-12b-L32-av</code> verbalizer.
          A Gemma Scope 2 SAE at L31 runs as a secondary instrument with auto-interp
          labels from Neuronpedia (~50% coverage on actively-firing features). No SGLang,
          no Anthropic API for the readout itself — just transformers, MPS, and the kitft
          inference recipe adapted to{" "}
          <code className="font-mono">model.generate(inputs_embeds=…)</code>.
        </p>
      </div>
    </div>
  );
}
