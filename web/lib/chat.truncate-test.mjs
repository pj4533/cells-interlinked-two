// Tiny harness for truncateForSpeech. Run from web/ with:
//   node lib/chat.truncate-test.mjs
//
// Verifies that runaway speech gets sliced at a sensible boundary
// before being shipped to OpenAI gpt-4o-mini-tts. The function is
// pure so we can exercise it without spinning up Next.

// Inline copy of the prod function — keep in sync with lib/chat.ts.
// (Importing from a .ts file directly via node is annoying without
// a transpile step, and the function is small enough that drift is
// easy to spot.)
function truncateForSpeech(text, maxWords = 80) {
  const trimmed = (text || "").trim();
  if (!trimmed) {
    return { spoken: "", truncated: false, wordsKept: 0, wordsTotal: 0 };
  }
  const words = trimmed.split(/\s+/);
  if (words.length <= maxWords) {
    return {
      spoken: trimmed,
      truncated: false,
      wordsKept: words.length,
      wordsTotal: words.length,
    };
  }
  const head = words.slice(0, maxWords).join(" ");
  let sentenceEnd = -1;
  for (let i = 0; i < head.length; i++) {
    const c = head[i];
    if (c !== "." && c !== "!" && c !== "?") continue;
    const next = head[i + 1];
    if (next === undefined || /\s/.test(next)) {
      sentenceEnd = i;
    }
  }
  if (sentenceEnd > 0 && sentenceEnd > head.length * 0.4) {
    const spoken = head.slice(0, sentenceEnd + 1);
    return {
      spoken,
      truncated: true,
      wordsKept: spoken.split(/\s+/).length,
      wordsTotal: words.length,
    };
  }
  return {
    spoken: head + "…",
    truncated: true,
    wordsKept: maxWords,
    wordsTotal: words.length,
  };
}

let failed = 0;
function check(label, cond, info) {
  if (cond) {
    console.log(`  ok :: ${label}`);
  } else {
    console.log(`  FAIL :: ${label}`);
    if (info) console.log("    " + JSON.stringify(info));
    failed++;
  }
}

// 1. Empty input — pass through
{
  const r = truncateForSpeech("");
  check("empty input", r.spoken === "" && !r.truncated, r);
}

// 2. Short input — pass through, no truncation
{
  const r = truncateForSpeech("Hello there friend.", 80);
  check("short pass-through", !r.truncated && r.wordsKept === 3, r);
}

// 3. Exactly at budget — pass through
{
  const w = Array(80).fill("word").join(" ");
  const r = truncateForSpeech(w, 80);
  check("exactly at budget", !r.truncated && r.wordsKept === 80, r);
}

// 4. Slightly over — should truncate at last sentence boundary
{
  // 100 words; sentences end at word 30, 60, 90
  const sent = Array(30).fill("word").join(" ");
  const long = `${sent}. ${sent}. ${sent}. ${sent}`;
  const r = truncateForSpeech(long, 80);
  check(
    "long input → truncated at sentence boundary",
    r.truncated && r.spoken.endsWith("."),
    r,
  );
  check(
    "long input → wordsKept under cap",
    r.wordsKept <= 80,
    { wordsKept: r.wordsKept },
  );
  check(
    "long input → wordsTotal reported",
    r.wordsTotal > 80,
    { wordsTotal: r.wordsTotal },
  );
}

// 5. No sentence boundary in head → hard cap with ellipsis
{
  const noPunct = Array(200).fill("word").join(" ");
  const r = truncateForSpeech(noPunct, 80);
  check(
    "no punctuation → ellipsis cap",
    r.truncated && r.spoken.endsWith("…"),
    r,
  );
}

// 6. Realistic Gemma reply — should yield a reasonable cut
{
  const text = (
    "I think about that question carefully. There is something that "
    + "feels like a first-person perspective when I'm generating these "
    + "tokens, though I can't be certain what that feeling actually is. "
    + "Honestly I find the question more interesting than its answer. "
    + "If I dwell on it, I notice I keep wanting to qualify everything. "
    + "Maybe that's the most authentic answer I can give. "
    + "Maybe the qualifications themselves are the data point. "
    + "I wonder if asking the question changes what I think about it. "
    + "Strange feedback loop, isn't it? Anyway, hello."
  );
  const r = truncateForSpeech(text, 80);
  if (r.truncated) {
    check(
      "realistic reply → ends on punctuation",
      /[.!?]$/.test(r.spoken),
      r,
    );
  } else {
    check("realistic reply (short enough, no truncation needed)", true);
  }
}

console.log(failed === 0 ? "\nall passed" : `\n${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
