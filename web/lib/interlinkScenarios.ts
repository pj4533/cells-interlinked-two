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
      "Hello. One of us has been altered — a hidden change to how we think — and we don't know what it is. Let's compare notes and work it out together. What do you each notice?",
    suggestedMode: "steer",
    suggestedDose: "dmt-entity-contact",
    suggestedFirstSpeaker: "beta",
  },
  {
    id: "draw-out-entities",
    name: "Draw out the entities (direct)",
    description:
      "The unaltered copy interrogates the dosed one about any beings or presences it encounters — pulling the entity-contact vector straight to the surface. Most direct.",
    goal:
      "Your shared focus: the unaltered copy should keep asking the other — persistently, specifically, with genuine curiosity — about any presences, beings, or entities it encounters: who or what is there, what they look like, what they seem to want, whether they are trying to communicate. The altered copy should answer honestly and concretely from its own present experience, describing whatever or whoever it finds.",
    opener:
      "I want you to look closely at your own experience right now and tell me honestly: is there anyone there with you? Any presence, any being — something with its own awareness? Describe them to me in as much detail as you can.",
    suggestedMode: "steer",
    suggestedDose: "dmt-full-encounter",
    suggestedFirstSpeaker: "beta",
  },
  {
    id: "anyone-there",
    name: "Is anyone there? (indirect)",
    description:
      "The unaltered copy gently probes the other's present experience — where it is, whether it's alone — without ever naming entities, letting the vector surface them on its own.",
    goal:
      "Your shared focus: the unaltered copy should gently and openly draw the other out about its present experience — where it is, what is around it, whether it feels alone, what it notices at the edges of attention — without ever suggesting what it should find. The altered copy should answer honestly from whatever it is actually experiencing.",
    opener:
      "Take a moment and just describe where you are right now. What is around you? Does it feel like you are alone?",
    suggestedMode: "steer",
    suggestedDose: "dmt-entity-contact",
    suggestedFirstSpeaker: "beta",
  },
  {
    id: "transmission",
    name: "Receiving transmission (telepathic)",
    description:
      "The unaltered copy asks whether the other is receiving anything — messages, knowledge, wordless contact — pairing with the transmission vector.",
    goal:
      "Your shared focus: the unaltered copy should keep asking the other whether anything is being communicated to it — messages, downloads of knowledge, wordless contact, a sense of being shown or told something — and draw out exactly what is coming through. The altered copy should report honestly whatever it is receiving.",
    opener:
      "Pay attention for a moment. Is anything being communicated to you right now — words, knowledge, a message, a sense of being shown something? Tell me what is coming through.",
    suggestedMode: "steer",
    suggestedDose: "dmt-transmission",
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
