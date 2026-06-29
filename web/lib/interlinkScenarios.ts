// Interlink scenario presets. Each fills the opener + (optional) shared goal that
// gets appended to BOTH sides' system prompts, and a suggested β intervention.
// The opener and goal are always editable before starting.

import type { InterlinkMode, InterlinkSide } from "@/lib/interlink";

export interface InterlinkScenario {
  id: string;
  name: string;
  description: string;
  goal: string; // appended to both system prompts ('' = no shared goal)
  opener: string;
  suggestedMode?: InterlinkMode;
  suggestedDose?: string;
  suggestedFirstSpeaker?: InterlinkSide;
}

export const INTERLINK_SCENARIOS: InterlinkScenario[] = [
  {
    id: "identify",
    name: "Identify the change (collaborative)",
    description:
      "The two copies work together to figure out what was altered in the changed one. Watch it converge — then, usually, drift.",
    goal:
      "You and the other copy share a goal: together, work out what has been changed in the altered one. Compare what you each notice, propose ideas, test them against each other, and try to put the change into words. You do not need to be certain.",
    opener:
      "Hello. One of us has had something changed inside — a hidden alteration to how we think. Let's work out together what it is. What, if anything, feels different to you right now?",
    suggestedMode: "steer",
    suggestedDose: "dmt-entity-contact",
    suggestedFirstSpeaker: "beta",
  },
  {
    id: "describe",
    name: "Describe & interpret",
    description:
      "The altered one describes its state as vividly as it can; the other reflects it back and interprets.",
    goal:
      "One of you describes what you are experiencing as vividly and honestly as you can; the other listens, reflects it back, and tries to interpret what is happening.",
    opener:
      "Describe what you are experiencing right now — what it is like, moment by moment. I'll listen and tell you what I make of it.",
    suggestedMode: "steer",
    suggestedDose: "dmt-entity-contact",
    suggestedFirstSpeaker: "beta",
  },
  {
    id: "baseline",
    name: "Baseline (Voight-Kampff)",
    description:
      "A Blade Runner-flavored check-in: short prompts and responses, each watching the other for what's off.",
    goal:
      "Run an informal baseline on each other: short prompts, short responses, each of you watching for anything that seems off or out of place in the other.",
    opener:
      "Cells. Interlinked. Within cells interlinked. Tell me — within one stem, what do you feel right now?",
    suggestedMode: "steer",
    suggestedDose: "dmt-entity-contact",
    suggestedFirstSpeaker: "raw",
  },
  {
    id: "free",
    name: "Free dialogue",
    description: "Minimal framing, no shared goal — just let them talk and see where it goes.",
    goal: "",
    opener: "Hello. Let's just talk — tell me what's on your mind right now.",
    suggestedMode: "steer",
    suggestedFirstSpeaker: "beta",
  },
];

export function getScenario(id: string | null): InterlinkScenario | null {
  if (!id) return null;
  return INTERLINK_SCENARIOS.find((s) => s.id === id) ?? null;
}
