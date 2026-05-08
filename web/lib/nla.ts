// Helpers for distilling the AV's verbose 2-3-paragraph output into a
// one- or two-line per-position diff. The AV always emits ~the same
// three-part shape because that's how it was trained:
//
//   §1  format / scaffold       ("Structured explanation format with...")
//   §2  surrounding-clause      ("The sentence 'X' mirrors the prompt...")
//   §3  this-token's-role       ("Final token 'When' repeats the opening...")
//
// Adjacent residuals share §1 and §2 almost verbatim; §3 is the only
// per-position-specific paragraph. The compact view surfaces just §3 with
// the redundant lead-in stripped ("Final token 'X' is/begins/marks ...").
//
// The full view (the original) stays one click away.

export interface NLAParts {
  /** §3 — the per-position-specific clause, with redundant lead-in trimmed. */
  role: string;
  /** §2 — surrounding-clause description. */
  context: string;
  /** §1 — overall format / scaffold description. */
  format: string;
}

const ROLE_LEADER = /^Final\s+token\s+(?:["'][^"']*["']\s+)?(?:is|begins|opens|ends|continues|repeats|completes|signals|marks|introduces|starts|closes|appears|sits|forms|completes)\s+/i;

export function splitNLA(nla: string | undefined | null): NLAParts {
  if (!nla) return { role: "", context: "", format: "" };
  const parts = nla.split(/\n\n+/).map((s) => s.trim()).filter(Boolean);
  let role = "";
  let context = "";
  let format = "";
  for (const part of parts) {
    if (!role && /^Final\s+token\b/i.test(part)) {
      role = part;
    } else if (
      !context &&
      /^(The\s+(sentence|phrase|text|token|word|opening|next|preceding|preceding|previous)|This\s+(sentence|phrase))\b/i.test(
        part,
      )
    ) {
      context = part;
    } else if (!format) {
      format = part;
    }
  }
  // Fallbacks if the structure was unexpected:
  if (!role && parts.length > 0) role = parts[parts.length - 1];
  if (!format && parts.length > 1) format = parts[0];

  // Strip the redundant lead-in. The token is already in its own column.
  role = role.replace(ROLE_LEADER, "").trim();
  // Drop a leading lowercase article/conjunction left dangling by the strip.
  role = role.replace(/^(?:and|but|then|now)\s+/i, "");
  // Capitalize the first character if it got lowercased mid-sentence.
  if (role.length > 0) {
    role = role[0].toUpperCase() + role.slice(1);
  }
  return { role, context, format };
}
