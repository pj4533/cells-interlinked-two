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
