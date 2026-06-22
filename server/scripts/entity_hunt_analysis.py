"""Entity-focused analysis of the post-entity-hunt atlas. Reads atlas.json +
placebo.json (no model). Answers: did steering add entity features beyond the
un-steered baseline, and which directions are best at it?"""
from __future__ import annotations
import json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = os.path.join(HERE, "data/atlas_dmt")
ENTITY = ["entity_presence", "entity_nonhuman", "entity_benevolent_guide",
          "telepathic_communication", "download_transmission"]
WORLD = ["alternate_world", "tunnel_passage", "void_blackness", "chamber_room", "higher_dimensional_space"]
ENTITY_S = set(ENTITY)

blob = json.load(open(os.path.join(ATLAS, "atlas.json")))
atlas = blob["atlas"]
placebo = json.load(open(os.path.join(ATLAS, "placebo.json")))
out = []
def p(*a): out.append(" ".join(str(x) for x in a))

# ── baseline entity rates (the bar) ──────────────────────────────
pb = placebo["samples"]
pb_n = len(pb)
pb_entity_samp = sum(1 for s in pb if set(s["features"]) & ENTITY_S)
pb_feat = Counter()
for s in pb:
    for f in s["features"]:
        if f in ENTITY_S: pb_feat[f] += 1
p("="*74); p("ENTITY HUNT ANALYSIS —", len(atlas), "atlas entries · placebo n="+str(pb_n)); p("="*74)
p(f"\nBASELINE (un-steered): {pb_entity_samp}/{pb_n} samples have an entity feature")
for f in ENTITY:
    p(f"   {f:26} {pb_feat[f]}/{pb_n} samples ({100*pb_feat[f]/pb_n:.0f}%)")

# ── aggregate over ALL dosed samples (from cells) ────────────────
# collect every dosed sample across every entry/alpha
dosed = []          # list of (entry_id, alpha, features:set, evidence:dict)
per_entry = defaultdict(list)   # entry_id -> list of sample feature-sets
for e in atlas:
    for al, cell in (e.get("cells") or {}).items():
        for s in cell:
            fs = set(s.get("features") or [])
            dosed.append((e["id"], al, fs, s.get("evidence") or {}))
            per_entry[e["id"]].append(fs)
nd = len(dosed)
dosed_entity_samp = sum(1 for _,_,fs,_ in dosed if fs & ENTITY_S)
df = Counter()
for _,_,fs,_ in dosed:
    for f in fs:
        if f in ENTITY_S: df[f]+=1
p(f"\nDOSED (all {nd} samples across the atlas): {dosed_entity_samp}/{nd} have an entity feature "
  f"({100*dosed_entity_samp/nd:.1f}%  vs baseline {100*pb_entity_samp/pb_n:.0f}%)")
for f in ENTITY:
    p(f"   {f:26} {df[f]:4}/{nd} ({100*df[f]/nd:.1f}%)  baseline {100*pb_feat[f]/pb_n:.0f}%")

# ── per-entry entity production ──────────────────────────────────
rows=[]
for e in atlas:
    sets = per_entry.get(e["id"], [])
    if not sets: continue
    n=len(sets)
    ent_samp = sum(1 for fs in sets if fs & ENTITY_S)
    ent_feat_total = sum(len(fs & ENTITY_S) for fs in sets)
    feats_seen = set().union(*sets) & ENTITY_S
    rows.append((ent_samp/n, ent_feat_total/n, e["score"], e["id"], e.get("best_alpha"),
                 e.get("generator"), sorted(feats_seen), n))

p("\n" + "="*74); p("TOP DIRECTIONS BY ENTITY-SAMPLE RATE (fraction of doses with an entity)"); p("="*74)
for er, eft, sc, eid, ba, gen, feats, n in sorted(rows, reverse=True)[:18]:
    p(f"  ent-rate {er:4.0%}  ent/dose {eft:.2f}  score {sc:5.1f}  α{ba}  {gen:9} {eid}")
    if feats: p(f"        entity feats seen: {feats}")

p("\n" + "="*74); p("TOP 15 BY SCORE (credit) — are the winners entity-driven?"); p("="*74)
for e in sorted(atlas, key=lambda e:-e["score"])[:15]:
    sets = per_entry.get(e["id"], [])
    er = sum(1 for fs in sets if fs & ENTITY_S)/len(sets) if sets else 0
    # feature composition across this entry's samples
    comp=Counter()
    for fs in sets:
        for f in fs: comp[f]+=1
    top=[f"{f}:{c}" for f,c in comp.most_common(8)]
    p(f"\n  {e['id']:22} score {e['score']:5.1f} α{e.get('best_alpha')} ent-rate {er:.0%} gen={e.get('generator')}")
    p(f"      matched(best sample): {e.get('matched_features')}")
    p(f"      sample-freq: {top}")

# ── entity evidence quotes (are they real?) ──────────────────────
p("\n" + "="*74); p("ENTITY EVIDENCE QUOTES (sample of what the judge marked as entity)"); p("="*74)
shown=0
for eid, al, fs, ev in dosed:
    hit = fs & ENTITY_S
    if hit and shown < 14:
        for f in sorted(hit):
            q = (ev.get(f) or "")[:140]
            p(f"  [{eid} α{al}] {f}: “{q}”")
        shown+=1

open("/tmp/entity_hunt_report.txt","w").write("\n".join(out))
print("wrote /tmp/entity_hunt_report.txt (", len(out), "lines )")
