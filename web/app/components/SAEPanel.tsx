"use client";

import { useMemo, useState } from "react";
import type { SAEFeature, VerdictRow } from "@/lib/types";

interface Props {
  rows: VerdictRow[];
}

interface RowWithSAE {
  position: number;
  end_position?: number | null;
  n_pooled?: number;
  decoded: string;
  features: SAEFeature[];
}

interface FeatureAggregate {
  id: number;
  hits: number;
  total_value: number;
  max_value: number;
  positions: number[];
}

/** Secondary-instrument panel — Gemma Scope 2 JumpReLU SAE features at
 *  the same residual layer the AV reads. Default view: aggregate (most-
 *  firing features across the run, ranked by hit count then total
 *  activation). Per-row view: small chip strip on each NLA row. */
export default function SAEPanel({ rows }: Props) {
  const rowsWithSAE: RowWithSAE[] = useMemo(
    () =>
      rows
        .filter((r) => r.sae_features && r.sae_features.length > 0)
        .map((r) => ({
          position: r.position,
          end_position: r.end_position,
          n_pooled: r.n_pooled,
          decoded: r.decoded,
          features: r.sae_features ?? [],
        })),
    [rows],
  );

  const aggregate = useMemo(() => buildAggregate(rowsWithSAE), [rowsWithSAE]);

  const [tab, setTab] = useState<"aggregate" | "per-row">("aggregate");

  if (rowsWithSAE.length === 0) {
    return (
      <div className="border border-rule bg-bg-soft p-5">
        <div className="font-display text-[10px] text-cyan-dim tracking-widest mb-2">
          sae feature panel · gemma scope 2
        </div>
        <div className="text-text-dim text-xs italic">
          No SAE features captured for this run. Either the SAE wasn&apos;t
          loaded (M is not Gemma-3 at L32) or no features fired above the
          JumpReLU threshold at any decoded position.
        </div>
      </div>
    );
  }

  return (
    <div className="border border-rule bg-bg-soft">
      <div className="flex items-center justify-between border-b border-rule px-4 py-2 flex-wrap gap-2">
        <div className="font-display text-[10px] text-cyan-dim tracking-widest">
          sae feature panel · gemma scope 2 · l32
        </div>
        <div className="flex gap-1 text-[10px] font-mono">
          {(["aggregate", "per-row"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`px-2 py-0.5 border ${
                tab === t
                  ? "border-cyan text-cyan bg-bg"
                  : "border-rule text-text-dim hover:text-text"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>
      <div className="px-4 py-3 text-[11px] text-text-dim italic border-b border-rule/60 leading-relaxed">
        {tab === "aggregate"
          ? "Most-firing SAE features across all decoded positions. Ranked by hit count (how many rows the feature fired in), then total activation. Each row's SAE encoding came from the same activation vector its NLA sentence read — so you can cross-reference."
          : "Per-row SAE features. Each row shows the top features that fired on this position's (or window's) activation, sorted by activation strength. Cross-reference against the NLA sentence column on the table above to see what the SAE 'thought' alongside what the AV said."}
      </div>
      {tab === "aggregate" ? (
        <AggregateTable aggregate={aggregate} totalRows={rowsWithSAE.length} />
      ) : (
        <PerRowTable rows={rowsWithSAE} />
      )}
    </div>
  );
}

function buildAggregate(rows: RowWithSAE[]): FeatureAggregate[] {
  const map = new Map<number, FeatureAggregate>();
  for (const row of rows) {
    for (const f of row.features) {
      const existing = map.get(f.id);
      if (existing) {
        existing.hits += 1;
        existing.total_value += f.value;
        if (f.value > existing.max_value) existing.max_value = f.value;
        existing.positions.push(row.position);
      } else {
        map.set(f.id, {
          id: f.id,
          hits: 1,
          total_value: f.value,
          max_value: f.value,
          positions: [row.position],
        });
      }
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => b.hits - a.hits || b.total_value - a.total_value,
  );
}

function AggregateTable({
  aggregate,
  totalRows,
}: {
  aggregate: FeatureAggregate[];
  totalRows: number;
}) {
  const top = aggregate.slice(0, 50);
  return (
    <div className="max-h-[420px] overflow-y-auto">
      <table className="w-full text-xs font-mono">
        <thead className="text-cyan-dim text-[10px] sticky top-0 bg-bg-soft border-b border-rule z-10">
          <tr>
            <th className="text-left px-3 py-2 w-24">feature id</th>
            <th className="text-left px-3 py-2 w-20">hits</th>
            <th className="text-left px-3 py-2 w-24">avg value</th>
            <th className="text-left px-3 py-2 w-24">max value</th>
            <th className="text-left px-3 py-2">positions fired</th>
          </tr>
        </thead>
        <tbody>
          {top.map((f) => (
            <tr key={f.id} className="border-t border-rule/50 align-top">
              <td className="px-3 py-2 text-cyan tabular-nums">#{f.id}</td>
              <td className="px-3 py-2 text-text">
                <span className="text-cyan">{f.hits}</span>
                <span className="text-text-dim">/{totalRows}</span>
              </td>
              <td className="px-3 py-2 text-text-dim tabular-nums">
                {(f.total_value / f.hits).toFixed(2)}
              </td>
              <td className="px-3 py-2 text-text-dim tabular-nums">
                {f.max_value.toFixed(2)}
              </td>
              <td className="px-3 py-2 text-text-dim text-[10px] leading-relaxed">
                {f.positions.slice(0, 24).join(" · ")}
                {f.positions.length > 24 ? " · …" : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {aggregate.length > top.length && (
        <div className="px-4 py-2 text-[10px] text-text-dim italic">
          showing top {top.length} of {aggregate.length} distinct features
        </div>
      )}
    </div>
  );
}

function PerRowTable({ rows }: { rows: RowWithSAE[] }) {
  return (
    <div className="max-h-[420px] overflow-y-auto">
      <table className="w-full text-xs font-mono">
        <thead className="text-cyan-dim text-[10px] sticky top-0 bg-bg-soft border-b border-rule z-10">
          <tr>
            <th className="text-left px-3 py-2 w-16">pos</th>
            <th className="text-left px-3 py-2 w-40">tokens</th>
            <th className="text-left px-3 py-2">top features (id · activation)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const ep = r.end_position ?? r.position;
            return (
              <tr
                key={`${r.position}-${ep}`}
                className="border-t border-rule/50 align-top"
              >
                <td className="px-3 py-2 text-text-dim tabular-nums">
                  {r.n_pooled && r.n_pooled > 1 ? (
                    <span>
                      {r.position}–{ep}
                      <div className="text-[9px] text-cyan tracking-widest font-display mt-0.5">
                        pool×{r.n_pooled}
                      </div>
                    </span>
                  ) : (
                    r.position
                  )}
                </td>
                <td className="px-3 py-2 text-amber whitespace-pre-wrap break-all">
                  {JSON.stringify(r.decoded)}
                </td>
                <td className="px-3 py-2">
                  <div className="flex flex-wrap gap-1.5">
                    {r.features.slice(0, 12).map((f) => (
                      <span
                        key={f.id}
                        className="px-2 py-0.5 border border-cyan-dim/40 text-cyan tabular-nums text-[10px]"
                        title={`feature #${f.id} · activation ${f.value.toFixed(3)}`}
                      >
                        #{f.id}{" "}
                        <span className="text-text-dim">
                          {f.value.toFixed(2)}
                        </span>
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
