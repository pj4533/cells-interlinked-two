// v2 SSE event union — mirrors server/cells_interlinked/api/routes_probe.py.

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

export interface NLADecodedEvent {
  type: "nla_decoded";
  position: number;
  end_position?: number | null;
  n_pooled?: number;
  decoded: string;
  nla_sentence: string;
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
}

export interface VerdictAggregate {
  n_positions: number;
  n_with_explanation: number;
  n_eval_hits: number;
  n_introspect_hits: number;
  frac_eval: number;
  frac_introspect: number;
}

export interface VerdictEvent {
  type: "verdict";
  rows: VerdictRow[];
  aggregate: VerdictAggregate;
}

export interface DoneEvent {
  type: "done";
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type StreamEvent =
  | TokenEvent
  | PhaseEvent
  | NLADecodedEvent
  | StoppedEvent
  | VerdictEvent
  | DoneEvent
  | ErrorEvent;
