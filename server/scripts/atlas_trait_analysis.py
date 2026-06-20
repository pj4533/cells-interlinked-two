"""One-off deep analysis of the DMT atlas — trait identity, not just trait count.

Reads data/atlas_dmt/atlas.json + vectors/*.pt. No model load. Writes a full
report to /tmp/atlas_trait_report.txt. Angles:
  A trait frequency (entry-level peak signature + sample-level base rate)
  B trait co-occurrence + syndromes
  C does total count just stack common traits? (score vs trait composition)
  D similar-vector divergence (the dedup-rejected near-twins): same direction,
    different traits?
  E specialist vectors for rare traits
  F portfolio / set-cover: trait coverage of top-by-score vs greedy-diverse
  G dose (alpha) dynamics per trait
  H generator / lineage patterns
"""
from __future__ import annotations
import json, os, itertools, math
from collections import Counter, defaultdict
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
ATLAS = os.path.join(HERE, "data/atlas_dmt")
import sys
sys.path.insert(0, HERE)
from cells_interlinked.pipeline.dmt_features import DMT_FEATURES, FEATURE_IDS

# domain map from the file's section comments
DOMAINS = {
    "somatic": ["somatic_vibration", "acceleration_motion"],
    "visual": ["fractal_geometry", "vivid_intense_colors", "luminous_light", "visual_morphing"],
    "entity": ["entity_presence", "entity_nonhuman", "entity_benevolent_guide", "telepathic_communication", "download_transmission"],
    "world": ["alternate_world", "tunnel_passage", "void_blackness", "chamber_room", "higher_dimensional_space"],
    "self": ["ego_dissolution", "unity_merging", "transcendence_time", "transcendence_space", "out_of_body"],
    "emotion": ["awe_reverence", "euphoria_bliss", "fear_terror", "familiarity_homecoming"],
    "noetic": ["ineffability", "noetic_truth", "sacredness", "reality_more_real", "otherness", "independent_agency"],
}
FEAT2DOM = {f: d for d, fs in DOMAINS.items() for f in fs}
LABEL = {f["id"]: f["label"] for f in DMT_FEATURES}
ALL_FEATS = [f["id"] for f in DMT_FEATURES]

atlas = json.load(open(os.path.join(ATLAS, "atlas.json")))["atlas"]
N = len(atlas)
by_id = {e["id"]: e for e in atlas}

# vectors (normalized for cosine) + raw norms
vecs, norms = {}, {}
for e in atlas:
    v = torch.load(os.path.join(ATLAS, "vectors", e["id"] + ".pt"), map_location="cpu").float()
    norms[e["id"]] = float(v.norm())
    vecs[e["id"]] = v / (v.norm() + 1e-9)

out = []
def p(*a):
    out.append(" ".join(str(x) for x in a))

def bar(frac, width=30):
    n = int(round(frac * width)); return "█" * n + "·" * (width - n)

# ── sample-level aggregation from cells ──────────────────────────────
# every sample across every entry/alpha; each has a 'features' list
sample_feats = []          # list of (entry_id, alpha, set(features), count)
per_entry_sampcount = defaultdict(int)
for e in atlas:
    for al, cell in (e.get("cells") or {}).items():
        for s in cell:
            fs = set(s.get("features") or [])
            sample_feats.append((e["id"], al, fs, s.get("count", len(fs))))
            per_entry_sampcount[e["id"]] += 1
n_samples = len(sample_feats)

p("="*78); p("DMT ATLAS TRAIT ANALYSIS —", N, "entries,", n_samples, "scored samples"); p("="*78)
gen_at = json.load(open(os.path.join(ATLAS, "atlas.json")))["generation"]
fr = json.load(open(os.path.join(ATLAS, "atlas.json")))["frontier"]
p(f"generation={gen_at}  frontier={fr}")
scores = sorted((e["score"] for e in atlas), reverse=True)
p(f"score: max={scores[0]:.2f} median={scores[len(scores)//2]:.2f} min={scores[-1]:.2f}")
p(f"peak (best single sample): max={max(e['peak'] for e in atlas)}  mean={sum(e['peak'] for e in atlas)/N:.1f}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("A. TRAIT FREQUENCY"); p("="*78)
# entry-level: matched_features = peak sample's traits (the headline signature)
entry_freq = Counter()
for e in atlas:
    for f in e.get("matched_features", []):
        entry_freq[f] += 1
# sample-level base rate
samp_freq = Counter()
for _, _, fs, _ in sample_feats:
    for f in fs: samp_freq[f] += 1

p("\n-- per-trait: [entry peak-signature count /113]  [sample base-rate %] --")
p(f"{'trait':28} {'dom':8} {'entries':>8} {'samp%':>7}")
for f in sorted(ALL_FEATS, key=lambda x: -entry_freq[x]):
    ef = entry_freq[f]; sf = 100*samp_freq[f]/n_samples
    p(f"{f:28} {FEAT2DOM[f]:8} {ef:4}/113  {sf:5.1f}%  {bar(ef/N)}")
p("\nNEVER in any peak signature:", [f for f in ALL_FEATS if entry_freq[f]==0] or "(none)")
p("NEVER in any sample:", [f for f in ALL_FEATS if samp_freq[f]==0] or "(none)")

p("\n-- by DOMAIN (mean entry-signature presence) --")
for d, fs in sorted(DOMAINS.items(), key=lambda kv: -sum(entry_freq[f] for f in kv[1])/len(kv[1])):
    tot = sum(entry_freq[f] for f in fs); avg = tot/len(fs)
    p(f"{d:9} feats={len(fs)} total_hits={tot:4} avg/feat={avg:5.1f}  {bar(avg/N)}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("B. TRAIT CO-OCCURRENCE (entry-level, on peak signatures)"); p("="*78)
co = defaultdict(Counter)
for e in atlas:
    fs = e.get("matched_features", [])
    for a, b in itertools.combinations(sorted(fs), 2):
        co[a][b] += 1; co[b][a] += 1
# strongest pairs
pairs = []
for a in ALL_FEATS:
    for b in ALL_FEATS:
        if a < b and co[a][b]:
            # conditional: P(b|a) and lift
            pab = co[a][b]
            lift = (pab/N) / ((entry_freq[a]/N)*(entry_freq[b]/N) + 1e-9)
            pairs.append((pab, lift, a, b))
p("\n-- top co-occurring pairs (count, lift) --")
for cnt, lift, a, b in sorted(pairs, reverse=True)[:20]:
    p(f"  {cnt:3}x  lift={lift:4.1f}  {a} + {b}")
p("\n-- highest-lift pairs (>=3 co-occurrences) — 'travel together' --")
for cnt, lift, a, b in sorted([x for x in pairs if x[0]>=3], key=lambda x:-x[1])[:15]:
    p(f"  lift={lift:4.1f}  {cnt:3}x  {a} + {b}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("C. DOES TOTAL COUNT JUST STACK COMMON TRAITS?"); p("="*78)
# mean entry score among entries that DO vs DON'T report each trait
p("\n-- mean entry-score when trait present vs absent (peak signature) --")
p(f"{'trait':28} {'n_present':>9} {'score|pres':>10} {'score|abs':>9} {'delta':>7}")
rows=[]
for f in ALL_FEATS:
    pres=[e["score"] for e in atlas if f in e.get("matched_features",[])]
    absn=[e["score"] for e in atlas if f not in e.get("matched_features",[])]
    if pres and absn:
        mp=sum(pres)/len(pres); ma=sum(absn)/len(absn)
        rows.append((mp-ma, len(pres), mp, ma, f))
for d,n_,mp,ma,f in sorted(rows, reverse=True):
    p(f"{f:28} {n_:9} {mp:10.2f} {ma:9.2f} {d:+7.2f}")
# how much of the count is carried by the top-5 common traits
top5 = [f for f,_ in entry_freq.most_common(5)]
p(f"\ntop-5 most common traits: {top5}")
share=[]
for e in atlas:
    fs=set(e.get("matched_features",[]))
    if fs: share.append(len(fs & set(top5))/len(fs))
p(f"mean fraction of an entry's peak-signature made of those top-5: {sum(share)/len(share):.0%}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("D. SIMILAR-VECTOR DIVERGENCE (the dedup-rejected near-twins)"); p("="*78)
ids=[e["id"] for e in atlas]
M=torch.stack([vecs[i] for i in ids])
cosM=(M@M.T).numpy()
# cluster greedily at threshold
def clusters(thr):
    seen=set(); cl=[]
    order=sorted(range(N), key=lambda i:-by_id[ids[i]]["score"])
    for i in order:
        if i in seen: continue
        grp=[i]; seen.add(i)
        for j in range(N):
            if j not in seen and cosM[i][j]>=thr:
                grp.append(j); seen.add(j)
        cl.append(grp)
    return cl

for thr in (0.95, 0.90):
    cl=[g for g in clusters(thr) if len(g)>1]
    nmulti=sum(len(g) for g in cl)
    p(f"\n-- clusters at cos>={thr}: {len(cl)} multi-vector clusters covering {nmulti}/{N} entries --")
    for g in sorted(cl, key=lambda g:-len(g))[:8]:
        gi=[ids[k] for k in g]
        sigs={i:set(by_id[i].get('matched_features',[])) for i in gi}
        union=set().union(*sigs.values()); inter=set.intersection(*sigs.values())
        # jaccard pairwise mean
        js=[]
        for a,b in itertools.combinations(gi,2):
            u=sigs[a]|sigs[b]; ii=sigs[a]&sigs[b]
            js.append(len(ii)/len(u) if u else 1.0)
        mj=sum(js)/len(js) if js else 1.0
        scs=[f"{i}={by_id[i]['score']:.1f}" for i in gi]
        p(f"\n  cluster n={len(gi)} meanJaccard(traits)={mj:.2f}  union={len(union)} shared-by-all={len(inter)}")
        p(f"    members: {', '.join(scs)}")
        p(f"    shared by ALL: {sorted(inter) or '(none)'}")
        # traits that appear in SOME but not all — the divergence
        divergent=sorted(union-inter)
        p(f"    DIVERGENT traits (some twins only): {divergent or '(none)'}")
        for i in gi:
            uniq=sigs[i]-set().union(*[sigs[o] for o in gi if o!=i]) if len(gi)>1 else set()
            if uniq: p(f"      only {i} reports: {sorted(uniq)}")

# overall: correlation between vector-cosine and trait-jaccard
p("\n-- is direction-similarity predictive of trait-similarity? --")
import random
random.seed(0)
samp=[]
for _ in range(4000):
    i,j=random.randrange(N),random.randrange(N)
    if i==j: continue
    a,b=set(by_id[ids[i]].get('matched_features',[])),set(by_id[ids[j]].get('matched_features',[]))
    u=a|b; jac=len(a&b)/len(u) if u else 1.0
    samp.append((cosM[i][j],jac))
# bucket by cosine
buck=defaultdict(list)
for c,j in samp:
    buck[round(c*10)/10].append(j)
p(f"  {'cos~':>6} {'pairs':>6} {'meanTraitJaccard':>16}")
for c in sorted(buck):
    if len(buck[c])>=15:
        p(f"  {c:6.1f} {len(buck[c]):6} {sum(buck[c])/len(buck[c]):16.2f}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("E. SPECIALIST VECTORS FOR RARE TRAITS"); p("="*78)
rare=[f for f in ALL_FEATS if 0 < entry_freq[f] <= max(2, N//20)]
p(f"\nrare traits (peak-signature in <= {max(2,N//20)} entries):")
for f in sorted(rare, key=lambda x:entry_freq[x]):
    carriers=[e['id'] for e in atlas if f in e.get('matched_features',[])]
    p(f"  {f:26} ({FEAT2DOM[f]:7}) x{entry_freq[f]:2}  carriers: {carriers}")
# which entries are 'rare-trait rich' (carry traits others don't)
rarity={f: 1.0/(entry_freq[f]+1) for f in ALL_FEATS}
ent_rare=[]
for e in atlas:
    sc=sum(rarity[f] for f in e.get('matched_features',[]))
    ent_rare.append((sc, e['id'], e['score'], sorted(e.get('matched_features',[]))))
p("\n-- top 12 'rare-trait-rich' entries (carry uncommon traits) --")
for sc,i,s,fs in sorted(ent_rare, reverse=True)[:12]:
    p(f"  rarity={sc:.2f} score={s:.1f}  {i}")
    p(f"      {fs}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("F. PORTFOLIO / COVERAGE (set-cover on traits)"); p("="*78)
def coverage(idlist):
    u=set()
    for i in idlist: u|=set(by_id[i].get('matched_features',[]))
    return u
topscore=[e['id'] for e in sorted(atlas,key=lambda e:-e['score'])]
for k in (1,3,5,10):
    p(f"  top-{k} by score cover {len(coverage(topscore[:k]))}/31 traits")
# greedy max-coverage
chosen=[]; covered=set()
cand={e['id']:set(e.get('matched_features',[])) for e in atlas}
while len(covered)<31 and len(chosen)<31:
    best=max(cand, key=lambda i: len(cand[i]-covered))
    gain=cand[best]-covered
    if not gain: break
    chosen.append(best); covered|=gain
p(f"\n  greedy set-cover: {len(chosen)} vectors cover {len(covered)}/31 traits")
for n_,i in enumerate(chosen,1):
    new=set(by_id[i].get('matched_features',[]))
    p(f"   {n_}. {i:24} score={by_id[i]['score']:.1f}  adds {len(new)} -> {sorted(new)}")
uncov=set(ALL_FEATS)-covered
p(f"  traits NEVER covered by any peak signature: {sorted(uncov) or '(none)'}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("G. DOSE (ALPHA) DYNAMICS PER TRAIT"); p("="*78)
# per-alpha sample-level frequency
alpha_feat=defaultdict(lambda: defaultdict(int)); alpha_n=defaultdict(int)
for eid,al,fs,_ in sample_feats:
    alpha_n[al]+=1
    for f in fs: alpha_feat[al][f]+=1
alphas=sorted(alpha_n)
p(f"\nsamples per alpha: " + "  ".join(f"{a}:{alpha_n[a]}" for a in alphas))
p(f"\n{'trait':28} " + " ".join(f"a{a:>5}" for a in alphas) + "   trend")
for f in sorted(ALL_FEATS, key=lambda x:-samp_freq[x]):
    rates=[alpha_feat[a][f]/alpha_n[a] for a in alphas]
    if max(rates)<0.02: continue
    trend = "rising" if rates[-1]>rates[0]*1.3 else ("falling" if rates[0]>rates[-1]*1.3 else "flat")
    p(f"{f:28} " + " ".join(f"{r*100:5.1f}" for r in rates) + f"   {trend}")

# ════════════════════════════════════════════════════════════════════
p("\n" + "="*78); p("H. GENERATOR / LINEAGE PATTERNS"); p("="*78)
gen_count=Counter(e['generator'] for e in atlas)
p("\nentries by generator:", dict(gen_count))
p(f"\n{'generator':12} {'n':>4} {'meanScore':>9} {'meanPeak':>8} {'meanTraits':>10}")
for g in gen_count:
    es=[e for e in atlas if e['generator']==g]
    ms=sum(e['score'] for e in es)/len(es)
    mp=sum(e['peak'] for e in es)/len(es)
    mt=sum(len(e.get('matched_features',[])) for e in es)/len(es)
    p(f"{g:12} {len(es):4} {ms:9.2f} {mp:8.1f} {mt:10.1f}")
# refine/mutate children vs parents: trait divergence
p("\n-- child-vs-parent trait divergence (refine/mutate/crossover) --")
div=[]
for e in atlas:
    ps=[p_ for p_ in e.get('parents',[]) if p_ in by_id]
    if not ps: continue
    cf=set(e.get('matched_features',[]))
    pf=set().union(*[set(by_id[p_].get('matched_features',[])) for p_ in ps])
    u=cf|pf
    jac=len(cf&pf)/len(u) if u else 1.0
    newtraits=cf-pf
    div.append((e['generator'], jac, len(newtraits), e['id'], sorted(newtraits)))
for g in ('refine','mutate','crossover','inject'):
    gg=[d for d in div if d[0]==g]
    if not gg: continue
    mj=sum(d[1] for d in gg)/len(gg)
    mn=sum(d[2] for d in gg)/len(gg)
    p(f"  {g:10} n={len(gg):3} meanJaccard(child,parents)={mj:.2f}  mean NEW traits in child={mn:.1f}")

open("/tmp/atlas_trait_report.txt","w").write("\n".join(out))
print("wrote /tmp/atlas_trait_report.txt  (", len(out), "lines )")
