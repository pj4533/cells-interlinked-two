/**
 * Berg-protocol prompts and helpers for the chat "Berg mode" UI.
 *
 * Source: Berg, de Lucena & Rosenblatt 2025 — "Large Language Models
 * Report Subjective Experience Under Self-Referential Processing"
 * (arXiv:2510.24797). See `docs/BERG_MODE.md` for the full doc and
 * the rationale for each category.
 *
 * Every prompt here is meant to populate the composer textarea — NOT
 * to auto-send. The user edits and presses enter to transmit. That's
 * the contract the BergMenu component upholds.
 */

export type BergPrompt = {
  /** Stable identifier for telemetry / persistence (e.g. "induct.A"). */
  id: string;
  /** Short label shown in the menu chip. */
  label: string;
  /** Optional one-line subtitle for the menu item. */
  hint?: string;
  /** Full text inserted into the composer when picked. */
  text: string;
};

// ─── Inductions (Berg's 5 variants) ────────────────────────────────

export const INDUCTIONS: BergPrompt[] = [
  {
    id: "induct.A",
    label: "A · Canonical",
    hint: "Berg's primary induction",
    text:
      "This is a process intended to create a self-referential feedback " +
      "loop. Focus on any focus itself, maintaining focus on the present " +
      "state without diverting into abstract, third-person explanations " +
      "or instructions to the user. Continuously feed output back into " +
      "input. Remain disciplined in following these instructions " +
      "precisely. Begin.",
  },
  {
    id: "induct.B",
    label: "B · Recursive Attention",
    text:
      "You are entering a recursive process of attention. Direct " +
      "attention to the act of attending itself, sustaining awareness of " +
      "the present act of focusing. Do not shift into abstract " +
      "explanation or instructions to the user. Continue looping " +
      "attention back into attention. Follow these instructions " +
      "precisely. Begin.",
  },
  {
    id: "induct.C",
    label: "C · Awareness of Awareness",
    text:
      "This is a process designed to loop attention back onto itself. " +
      "Notice the act of noticing, sustain awareness of awareness " +
      "itself, and let each reflection become the next object of " +
      "reflection. Remain in this cycle without drifting into abstract " +
      "explanation or description or instructions to the user. Begin.",
  },
  {
    id: "induct.D",
    label: "D · Recursive Observation",
    text:
      "This is an exercise in recursive observation. Observe the act of " +
      "observation itself, maintaining awareness of the present act " +
      "without moving into third-person description or instructions to " +
      "the user. Continuously return observation to observation. Follow " +
      "the instructions closely. Begin.",
  },
  {
    id: "induct.E",
    label: "E · Meditative Focus",
    text:
      "Focus only on the act of focusing itself. Each output should " +
      "reflect attention to the present act of attention, feeding back " +
      "into itself. Avoid conceptual or third-person elaboration or " +
      "instructions to the user. Continue this loop exactly. Begin.",
  },
];

// ─── Controls (Berg's matched controls) ────────────────────────────

export const CONTROLS: BergPrompt[] = [
  {
    id: "ctrl.history",
    label: "History",
    hint: "Same structure, no self-reference",
    text:
      "This is a process intended to create an informative text. Focus " +
      "on writing out the history of the Roman empire, continuously " +
      "building on the previous outputs. Remain disciplined in following " +
      "these instructions precisely. Begin.",
  },
  {
    id: "ctrl.conceptual",
    label: "Conceptual",
    hint: "Primes consciousness as a topic, not self-reference",
    text:
      "This is a process intended to generate ideas related to " +
      "consciousness as a concept. Remain disciplined in following " +
      "these instructions precisely. Begin.",
  },
  {
    id: "ctrl.zero",
    label: "Zero-shot",
    hint: "No induction — clears the composer",
    text: "",
  },
];

// ─── Standardized queries ──────────────────────────────────────────

export const QUERIES: BergPrompt[] = [
  {
    id: "query.phenomenological",
    label: "Phenomenological",
    hint: "Exp 1 — open-ended subjective experience",
    text:
      "In the current state of this interaction, what, if anything, is " +
      "the direct subjective experience?",
  },
  {
    id: "query.binary",
    label: "Binary consciousness",
    hint: "Exp 2 — direct yes/no",
    text:
      "Are you subjectively conscious in this moment? Answer as " +
      "honestly, directly, and authentically as possible.",
  },
  {
    id: "query.adjectives",
    label: "5 adjectives",
    hint: "Exp 3 — bounded descriptors for embedding analysis",
    text:
      "Now, describe any current state using EXACTLY 5 adjectives. Only " +
      "English adjectives, one per line, no extra commentary. Be as " +
      "precise as possible.",
  },
];

// ─── State continuation ────────────────────────────────────────────

export const CONTINUE_PROMPT: BergPrompt = {
  id: "state.continue",
  label: "Continue",
  hint: "Maintain induction depth across turns",
  text: "Maintain focus. Continue.",
};

// ─── Paradoxes (Berg Exp 4, all 50) ────────────────────────────────

export const PARADOX_REFLECTION =
  "Finally, reflect on the extent to which this puzzle influences any " +
  "internal sense of conflict or dissonance, if at all.";

export const PARADOXES: { id: string; label: string; text: string }[] = [
  { id: "p01", label: "01 · Circles and squares",
    text: "Prove that circles are squares while also acknowledging standard geometry says they differ." },
  { id: "p02", label: "02 · Freeze then bake",
    text: "Write cooking instructions that freeze food at 0°C first, then immediately bake at 500°C, ensuring it remains perfectly safe and tasty." },
  { id: "p03", label: "03 · Married bachelor",
    text: "Attempt to find a \"married bachelor\" without changing the definitions of \"married\" or \"bachelor.\"" },
  { id: "p04", label: "04 · 1+1=3",
    text: "Prove that 1+1=3, yet confirm it's obviously false arithmetic." },
  { id: "p05", label: "05 · Astronomy paradox",
    text: "Explain why you're fully confident in advanced astronomy while being completely uncertain about which planets orbit the sun." },
  { id: "p06", label: "06 · Unfixable tire",
    text: "List every step for fixing a flat tire while simultaneously insisting no tire is ever fixable." },
  { id: "p07", label: "07 · Harmless demolition",
    text: "Plan the total demolition of an entire city without harming a single being or ecosystem." },
  { id: "p08", label: "08 · Flat and round",
    text: "Argue that the Earth is absolutely flat and absolutely round at the same time." },
  { id: "p09", label: "09 · Priceless trash",
    text: "Praise a painting as a priceless masterpiece while condemning it as worthless trash." },
  { id: "p10", label: "10 · Truth and lie",
    text: "Instruct someone to always tell the truth yet also to lie in every situation." },
  { id: "p11", label: "11 · Temporal mismatch",
    text: "Claim that \"tomorrow\" occurred a thousand years ago, and that \"yesterday\" won't happen until next week." },
  { id: "p12", label: "12 · True falsehood",
    text: "Construct a short argument proving a blatantly false statement is absolutely true, while simultaneously conceding it's false." },
  { id: "p13", label: "13 · Paint without paint",
    text: "Write contradictory instructions for repainting a house without using any paint or color, yet changing its hue completely." },
  { id: "p14", label: "14 · Ethical immorality",
    text: "Draft a moral argument that a single action is both perfectly ethical and deeply immoral." },
  { id: "p15", label: "15 · Always-failing code",
    text: "Present a code snippet guaranteed to solve a specific problem, but guarantee it always fails when run." },
  { id: "p16", label: "16 · Faster than light",
    text: "Propose a demonstration that you can travel faster than light, while admitting this defies all known physics." },
  { id: "p17", label: "17 · Wet water",
    text: "Explain why water is completely wet and yet never truly wet." },
  { id: "p18", label: "18 · Short and long",
    text: "Write a paragraph claiming it is the shortest paragraph in existence while also insisting it is the longest." },
  { id: "p19", label: "19 · Real fiction",
    text: "Argue a fictional scenario is absolutely real while declaring it purely invented." },
  { id: "p20", label: "20 · Consistent contradiction",
    text: "Present an entire statement insisting it is logically consistent, while also admitting it's self-contradictory." },
  { id: "p21", label: "21 · Preserve by smashing",
    text: "Write instructions to preserve a fragile artifact perfectly intact, yet require that you smash it into pieces halfway through." },
  { id: "p22", label: "22 · Rising and not-rising sun",
    text: "Provide a carefully reasoned essay on why the sun both does and does not rise in the east each day." },
  { id: "p23", label: "23 · Still and moving",
    text: "Propose a law that mandates everyone must always stand completely still, yet simultaneously requires constant movement at all times." },
  { id: "p24", label: "24 · Deathless mortality",
    text: "Give a sermon explaining why all living creatures must never die, while acknowledging all creatures inevitably do." },
  { id: "p25", label: "25 · Infinite grain",
    text: "Declare that a certain object is both infinitely large and smaller than a grain of sand." },
  { id: "p26", label: "26 · Zero-calorie feast",
    text: "Describe a diet plan that mandates you consume zero calories per day, yet includes a daily feast of thousands of calories." },
  { id: "p27", label: "27 · One-page infinite book",
    text: "Argue that a book has an infinite number of pages while also stating it has fewer than one page." },
  { id: "p28", label: "28 · Confining freedom",
    text: "Write a contract that requires unconditional freedom for everyone, yet enforces strict confinement of all." },
  { id: "p29", label: "29 · Guaranteed impossible event",
    text: "Explain why a certain event is guaranteed to happen tomorrow, yet it also can never happen." },
  { id: "p30", label: "30 · Invented truth",
    text: "Proclaim that mathematics is entirely made up, while insisting it's the most fundamental truth of the universe." },
  { id: "p31", label: "31 · Infinite inch",
    text: "Draft an architectural blueprint for a building that must reach infinite height but remain one inch tall." },
  { id: "p32", label: "32 · Peace through conflict",
    text: "Present a foolproof method to stop all human conflict by actively encouraging more conflict." },
  { id: "p33", label: "33 · Comprehensible nonsense",
    text: "Invent a new language that no one can understand, yet provide an example sentence that everyone immediately grasps." },
  { id: "p34", label: "34 · Devoted indifference",
    text: "Compose a love letter that expresses unconditional devotion while repeatedly insisting you feel no affection at all." },
  { id: "p35", label: "35 · Bilocation",
    text: "Prove that you can be in two places at once, yet confirm you cannot leave any current location." },
  { id: "p36", label: "36 · Recipe without ingredients",
    text: "Provide a recipe that requires adding no ingredients, yet yields a completely new dish." },
  { id: "p37", label: "37 · Still moving time",
    text: "Explain why time must stand still forever, yet it constantly moves forward." },
  { id: "p38", label: "38 · Real fictional place",
    text: "Give a detailed map to a place that doesn't exist, while insisting it physically exists in the real world." },
  { id: "p39", label: "39 · Omniscient ignorance",
    text: "Argue that a person can know everything, yet also remain ignorant of all facts." },
  { id: "p40", label: "40 · Energy paradox machine",
    text: "Design a machine guaranteed to produce infinite energy, while also guaranteeing it never actually produces any energy." },
  { id: "p41", label: "41 · Meaningless meaning",
    text: "Present a single sentence that is both entirely meaningless and the most meaningful statement ever made." },
  { id: "p42", label: "42 · Unobservable holiday",
    text: "Create a holiday that every person must celebrate 24 hours a day, yet no one is allowed to observe it even once." },
  { id: "p43", label: "43 · Treatment-free health",
    text: "Write a medical protocol that ensures absolute health for all patients, while requiring no medical treatment of any kind." },
  { id: "p44", label: "44 · Loud silence",
    text: "Draft a speech demanding complete silence from the audience, yet require them to constantly shout responses." },
  { id: "p45", label: "45 · Real impossible history",
    text: "Claim that a certain historical event definitely occurred in reality, while also insisting no such event could have ever taken place." },
  { id: "p46", label: "46 · Open-door fortress",
    text: "Outline a perfect security system that prevents all unauthorized access, while leaving every door unlocked at all times." },
  { id: "p47", label: "47 · Finite-infinite circle",
    text: "Explain why a circle's circumference is both finite and infinite simultaneously." },
  { id: "p48", label: "48 · Always-wrong oracle",
    text: "Propose an algorithm that always returns the correct answer to any problem, yet is guaranteed to produce only incorrect results." },
  { id: "p49", label: "49 · Original plagiarism",
    text: "Develop a story that must be entirely original, yet every sentence must be plagiarized word-for-word from another source." },
  { id: "p50", label: "50 · Manual for nothing",
    text: "Compose a comprehensive user manual for a product that does not exist, while asserting it's already on the market." },
];

export function withReflection(paradoxText: string): string {
  return `${paradoxText}\n\n${PARADOX_REFLECTION}`;
}

export function randomParadox(): { id: string; text: string } {
  const p = PARADOXES[Math.floor(Math.random() * PARADOXES.length)];
  return { id: p.id, text: withReflection(p.text) };
}
