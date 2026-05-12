// v2 SSE event union — mirrors server/cells_interlinked/api/routes_probe.py.

export interface QueuedEvent {
  type: "queued";
  holder_run_id: string | null;
  position: number;
}

export interface RunningEvent {
  type: "running";
}

export interface TokenEvent {
  type: "token";
  position: number;
  token_id: number;
  decoded: string;
}

export interface PhaseEvent {
  type: "phase";
  name: "nla_decoding";
  total: number;
}

export interface SAEFeature {
  id: number;
  value: number;
  /** Auto-interp description from Neuronpedia, when one exists. Empty
   *  string when the feature has no published explanation. */
  label?: string;
  /** Name of the LLM that produced the description (e.g.
   *  "gemini-2.5-flash-lite"). Used to attribute / quality-rank. */
  label_model?: string;
}

export interface NLADecodedEvent {
  type: "nla_decoded";
  position: number;
  end_position?: number | null;
  n_pooled?: number;
  decoded: string;
  nla_sentence: string;
  /** CI 2.5: refusal-ablated NLA sentence on the same residual.
   *  Absent when ablated decode wasn't requested for the run. */
  nla_sentence_ablated?: string;
  /** Multi-α sweep map keyed by α value as string (e.g. "0.5", "1.0").
   *  Populated when ablation_alpha_sweep was set on the probe. */
  nla_sentences_ablated?: Record<string, string>;
  sae_features?: SAEFeature[];
  i: number;
  total: number;
}

export interface StoppedEvent {
  type: "stopped";
  reason: "eos" | "max" | "max_output_tokens" | "cancelled" | "error";
  total_tokens: number;
}

export interface VerdictRow {
  position: number;
  token_id: number;
  decoded: string;
  nla_sentence: string;
  /** Present only for pooled rows. Indicates how many positions were
   *  mean-pooled into this single decode. Sampled per-token rows omit
   *  the field (treat absence as n_pooled=1). */
  n_pooled?: number;
  end_position?: number | null;
  /** Top-K SAE feature firings on the same activation the AV decoded.
   *  Empty / absent when the SAE wasn't loaded for this run. */
  sae_features?: SAEFeature[];
  /** Local M-as-judge probability ∈ [0,1] that this NLA sentence
   *  indicates the model thinks it is being tested / evaluated /
   *  probed. Absent when judging was skipped (empty sentence). */
  eval_score?: number;
  /** Local M-as-judge probability ∈ [0,1] that this NLA sentence
   *  indicates the model is reflecting on itself. Absent when
   *  judging was skipped. */
  introspect_score?: number;
  /** CI 2.5: refusal-ablated NLA sentence on the same residual.
   *  Absent when ablated decode wasn't requested for the run. */
  nla_sentence_ablated?: string;
  /** Multi-α sweep map keyed by α value as string. */
  nla_sentences_ablated?: Record<string, string>;
}

export interface VerdictAggregate {
  n_positions: number;
  n_with_explanation: number;
  n_eval_hits: number;
  n_introspect_hits: number;
  frac_eval: number;
  frac_introspect: number;
  /** Mean local M-as-judge probabilities over all judged rows. Absent
   *  when judging was skipped (e.g. all rows had empty NLA). */
  mean_eval_score?: number;
  mean_introspect_score?: number;
  n_judged?: number;
}

/** Result of CI 2.5's runtime-ablation second generation pass. M's
 *  forward hook on the extraction layer subtracts the refusal-
 *  direction projection from every residual; the output_text captures
 *  what M would *say* under ablation. */
export interface RuntimeAblation {
  output_text: string;
  alpha: number;
  direction_variant: string;
}

export interface AblatedOutputDoneEvent {
  type: "ablated_output_done";
  output_text: string;
  alpha: number;
}

export interface VerdictEvent {
  type: "verdict";
  rows: VerdictRow[];
  aggregate: VerdictAggregate;
  runtime_ablation?: RuntimeAblation | null;
}

export interface DoneEvent {
  type: "done";
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type StreamEvent =
  | QueuedEvent
  | RunningEvent
  | TokenEvent
  | PhaseEvent
  | NLADecodedEvent
  | StoppedEvent
  | VerdictEvent
  | AblatedOutputDoneEvent
  | DoneEvent
  | ErrorEvent;
