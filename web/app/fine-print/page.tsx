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
        <p>
          Locally, the readout runs on Apple Silicon — a Mac Studio M2 Ultra, 64GB unified
          memory, MPS backend. The default base model is Qwen2.5-7B-Instruct paired with
          kitft&apos;s nla-qwen2.5-7b-L20-av; the design doc&apos;s primary target is
          Gemma-3-12B-IT + nla-gemma3-12b-L32-av (also fits at bf16). No SGLang, no Anthropic
          API for the readout itself — just transformers, MPS, and the kitft inference recipe
          adapted to <code className="font-mono">model.generate(inputs_embeds=…)</code>.
        </p>
      </div>
    </div>
  );
}
