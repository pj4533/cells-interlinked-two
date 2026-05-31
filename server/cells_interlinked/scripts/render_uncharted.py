"""Render the uncharted activation-states some way OTHER than text.

Loads data/uncharted_traj.pt (from capture_uncharted_traj.py) and renders THREE
honest, non-linguistic visual encodings of the captured L32 trajectories. The
discipline: every visual parameter is driven by a REAL measured quantity — none
of it is decorative noise. Beauty as a faithful encoding.

Per run we measure (decoder-free, no language model in the loop):
  - covariance eigenspectrum  λ_1..λ_k      (the Trip View "truth anchor")
  - participation ratio (eff-dim)           = (Σλ)² / Σλ²
  - top-3 PCA coords of the trajectory      (the 3-of-3840 shadow)

Three styles, each a different faithful mapping:
  1. SPECTRAL MANDALA   — eigenvalues drive the amplitudes of angular harmonics
     of a radial curve. eff-dim → lobe complexity. A near-1D state is a smooth
     ring; a high-eff-dim state is a many-lobed rosette. The SHAPE *is* the
     spectrum. Hue = eff-dim (relative).
  2. CYMATIC FIELD      — the spectrum synthesised as a 2D standing-wave
     interference pattern (Chladni-like). Amplitudes from real eigenvalues;
     more spectral spread → more fringes. Each state's spectral fingerprint.
  3. NEBULA             — the literal 3D PCA point cloud, rendered as a soft
     additive-glow volume (the current dots, but luminous) with the trajectory
     thread drawn through it. Tight compact glow = a real attractor; diffuse
     smear = drift. Colour = position along the path (flow).

Output: PNG contact sheets in /tmp/uncharted_viz/ — one per style, all 6
conditions in a grid, so you can see both the aesthetic AND that the states are
distinct + (across the two prompts) reproducible.

No model needed; run anytime:
    cd server && uv run python -m cells_interlinked.scripts.render_uncharted
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .mpa_probe import DATA

OUT = Path("/tmp/uncharted_viz")
OUT.mkdir(parents=True, exist_ok=True)

S = 460          # per-cell render size
BG = np.array([8, 6, 16], np.float32) / 255.0

# dose-aesthetic palette: teal -> blue -> violet -> magenta -> amber
_STOPS = [
    (0.00, (0, 205, 180)),
    (0.30, (45, 120, 255)),
    (0.55, (150, 80, 255)),
    (0.78, (255, 70, 200)),
    (1.00, (255, 185, 95)),
]


def palette(t):
    """t in [0,1] -> rgb float 0..1 (vectorised over np array)."""
    t = np.clip(t, 0, 1)
    xs = np.array([s[0] for s in _STOPS])
    cols = np.array([s[1] for s in _STOPS], np.float32) / 255.0
    out = np.empty(t.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, xs, cols[:, c])
    return out


def blur_f(arr, sigma):
    """Gaussian-blur a float [H,W] image, HDR-safe (FFT; PIL lacks F-mode blur)."""
    h, w = arr.shape
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * g)).astype(np.float32)


def bloom(img):
    """img [H,W,3] HDR float -> tonemapped 0..1 with additive glow."""
    glow = np.zeros_like(img)
    for rad, w in ((3, 0.5), (9, 0.35), (24, 0.25), (60, 0.18)):
        for c in range(3):
            glow[..., c] += w * blur_f(np.ascontiguousarray(img[..., c]), rad)
    hdr = img + glow
    tm = hdr / (1.0 + hdr)                      # Reinhard tonemap
    out = np.power(np.clip(tm, 0, 1), 1 / 2.2)  # gamma
    return BG * (1 - out.max(-1, keepdims=True).clip(0, 1)) * 0.0 + out + BG  # lift blacks to bg


def _splat(buf, xs, ys, rgb, amp):
    """Additive point splats into HDR buf [H,W,3] (1px; bloom does the glow)."""
    h, w = buf.shape[:2]
    xi = np.round(xs).astype(int); yi = np.round(ys).astype(int)
    m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi, yi = xi[m], yi[m]
    col = rgb[m] if rgb.ndim == 2 else np.broadcast_to(rgb, (xi.size, 3))
    a = amp[m] if np.ndim(amp) else np.full(xi.size, amp, np.float32)
    np.add.at(buf, (yi, xi), col * a[:, None])


# ── geometry ────────────────────────────────────────────────────────────────
def geometry(traj):
    """traj [N,D] -> dict(eig=λ normalised desc, effdim, pca3 [N,3])."""
    # Gemma L32 has a few massive-activation outlier dims that dwarf everything
    # else; drop the top-magnitude dims so the geometry reflects the trajectory,
    # not the constant outliers (which carry no per-token structure).
    keep = np.argsort(-np.abs(traj).mean(0))[8:]
    traj = traj[:, keep]
    X = traj - traj.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lam = (s ** 2)
    lam = lam / (lam.sum() + 1e-12)
    effdim = (lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)
    pca3 = U[:, :3] * s[:3]
    return {"eig": lam, "effdim": float(effdim), "pca3": pca3}


# ── style 1: spectral mandala ────────────────────────────────────────────────
def mandala(g, hue_t, H=16, sym=6):
    buf = np.zeros((S, S, 3), np.float32)
    cx = cy = S / 2
    a = np.sqrt(g["eig"][:H] / (g["eig"][0] + 1e-12))      # harmonic amplitudes (real)
    phi = 2 * np.pi * ((np.arange(H) * 0.6180339887) % 1)  # orientation only
    th = np.linspace(0, 2 * np.pi, 2400, endpoint=False)
    r = np.ones_like(th)
    for i in range(H):
        r += 0.55 * a[i] * np.cos((i + 1) * th + phi[i])
    r = r / (r.max() + 1e-9)
    R0 = S * 0.40
    base = palette(np.array([hue_t]))[0]
    for k in range(sym):
        rot = k * 2 * np.pi / sym
        xs = cx + R0 * r * np.cos(th + rot)
        ys = cy + R0 * r * np.sin(th + rot)
        # colour drifts along the curve (teal->hue) so lobes read as depth
        tcol = palette(0.15 + 0.7 * (0.5 + 0.5 * np.cos(2 * th)) * hue_t + 0.0 * th)
        col = 0.5 * tcol + 0.5 * base
        _splat(buf, xs, ys, col.astype(np.float32), 0.9)
    return buf


# ── style 2: cymatic interference field ──────────────────────────────────────
def cymatic(g, hue_t, H=18):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    x = (xx / S - 0.5) * 2; y = (yy / S - 0.5) * 2
    a = np.sqrt(g["eig"][:H])
    a = a / (a[0] + 1e-12)
    field = np.zeros((S, S), np.float32)
    for i in range(H):
        ang = i * 2.399963                                  # golden-angle spread
        f = 2.2 * (i + 1)                                   # harmonic freq = PC index
        field += a[i] * np.cos(f * (math.cos(ang) * x + math.sin(ang) * y) + 1.7 * i)
    field = field / (np.abs(field).max() + 1e-9)
    inten = np.abs(field) ** 1.6                            # antinodal brightness
    t = 0.2 + 0.6 * hue_t + 0.2 * field                     # hue from eff-dim + field sign
    img = palette(t.ravel()).reshape(S, S, 3) * inten[..., None]
    return img.astype(np.float32) * 2.2


# ── style 3: nebula (literal PCA cloud, luminous) ────────────────────────────
def nebula(g):
    buf = np.zeros((S, S, 3), np.float32)
    p = g["pca3"].astype(np.float32)
    if len(p) < 2:
        return buf
    span = np.abs(p[:, :2]).max() + 1e-6
    sc = S * 0.40 / span
    px = S / 2 + p[:, 0] * sc
    py = S / 2 + p[:, 1] * sc
    tcol = palette(np.linspace(0, 1, len(p)))               # colour = flow (time)
    depth = (p[:, 2] - p[:, 2].min()) / (np.ptp(p[:, 2]) + 1e-6)
    # trajectory thread (dense interpolation between consecutive tokens)
    for i in range(len(p) - 1):
        n = 24
        xs = np.linspace(px[i], px[i + 1], n); ys = np.linspace(py[i], py[i + 1], n)
        col = 0.5 * (tcol[i] + tcol[i + 1])
        _splat(buf, xs, ys, np.broadcast_to(col, (n, 3)).copy(), 0.10)
    # token nodes (brighter, size via depth)
    for i in range(len(p)):
        rad = 1 + 4 * depth[i]
        ang = np.linspace(0, 2 * np.pi, 40)
        for rr in np.linspace(0.3, rad, 4):
            _splat(buf, px[i] + rr * np.cos(ang), py[i] + rr * np.sin(ang),
                   np.broadcast_to(tcol[i], (40, 3)).copy(), 0.5 / (1 + rr))
    return buf


# ── shared embedding: each condition's direction signature ───────────────────
def shared_embedding(runs, k=6):
    """Common frame from the 6 condition centroids. Returns per-condition
    signature (centroid coords in top-k PCs) — the direction's identity."""
    base = runs["baseline|p0"]["traj"].float().numpy()
    drop = np.argsort(-np.abs(base).mean(0))[:8]
    mask = np.ones(base.shape[1], bool); mask[drop] = False
    trajs = {key: runs[f"{key}|p0"]["traj"].float().numpy()[:, mask] for key in ORDER}
    cents = np.stack([trajs[key].mean(0) for key in ORDER], 0)
    gm = cents.mean(0)
    _, _, Vt = np.linalg.svd(cents - gm, full_matrices=False)
    basis = Vt[:k]
    sig = {key: (trajs[key].mean(0) - gm) @ basis.T for key in ORDER}
    proj = {key: (trajs[key] - gm) @ basis.T for key in ORDER}
    return sig, proj, basis, gm, mask


# ── style 6: signature mandala (spectrum -> shape, DIRECTION -> fingerprint) ──
def mandala_sig(g, sig, H=13, sym=5):
    """Amplitudes from the eigenspectrum (complexity = eff-dim); a SECOND set of
    overtones + the hue come from the direction signature -> distinct per state,
    still a mandala. Same direction -> same fingerprint (reproducible)."""
    buf = np.zeros((S, S, 3), np.float32)
    cx = cy = S / 2
    a = np.sqrt(g["eig"][:H] / (g["eig"][0] + 1e-12))     # spectral (real)
    phi = 2 * np.pi * ((np.arange(H) * 0.6180339887) % 1)
    s = np.asarray(sig, np.float32)
    sn = s / (np.abs(s).max() + 1e-9)                     # normalised signature (real)
    overn = np.array([2, 3, 5, 7, 11, 13])[:len(sn)]      # directional overtones
    hue_t = (math.atan2(float(sn[1]), float(sn[0])) / (2 * math.pi)) % 1.0  # hue = direction angle
    twist = float(np.tanh(sn.sum())) * 1.6                # chirality from signed signature
    th = np.linspace(0, 2 * np.pi, 2600, endpoint=False)
    r = np.ones_like(th)
    for i in range(H):
        r += 0.50 * a[i] * np.cos((i + 1) * th + phi[i])
    for j in range(len(sn)):                               # directional fingerprint
        r += 0.40 * abs(sn[j]) * np.cos(overn[j] * th + np.pi * (sn[j] < 0) + j)
    r = r / (r.max() + 1e-9)
    R0 = S * 0.40
    base = palette(np.array([hue_t]))[0]
    for k in range(sym):
        rot = k * 2 * np.pi / sym
        spin = twist * (r - r.mean())                      # radius-dependent swirl
        xs = cx + R0 * r * np.cos(th + rot + spin)
        ys = cy + R0 * r * np.sin(th + rot + spin)
        tcol = palette((hue_t + 0.18 * np.cos(3 * th)) % 1.0)
        col = (0.55 * tcol + 0.45 * base).astype(np.float32)
        _splat(buf, xs, ys, col, 0.9)
    return buf


# ── style 4: shared-space manifold + rays ────────────────────────────────────
def shared_map(runs, big=920):
    """One image, common embedding: affective cloud (baseline+named) at the core,
    each uncharted state as a distinct ray flying off-manifold in its own
    direction. The honest 'what distinguishes them' picture (it's direction)."""
    base = runs["baseline|p0"]["traj"].float().numpy()
    drop = np.argsort(-np.abs(base).mean(0))[:8]
    mask = np.ones(base.shape[1], bool); mask[drop] = False
    trajs = {k: runs[f"{k}|p0"]["traj"].float().numpy()[:, mask] for k in ORDER}
    cents = np.stack([trajs[k].mean(0) for k in ORDER], 0)
    gm = cents.mean(0)
    _, _, Vt = np.linalg.svd(cents - gm, full_matrices=False)
    basis = Vt[:3]
    P = {k: (trajs[k] - gm) @ basis.T for k in ORDER}
    allp = np.concatenate(list(P.values()), 0)
    sc = big * 0.42 / (np.abs(allp[:, :2]).max() + 1e-6)
    buf = np.zeros((big, big, 3), np.float32)
    ctr = np.array([big / 2, big / 2])
    for k in ORDER:
        xy = ctr + P[k][:, :2] * sc * np.array([1, -1])
        if k.startswith("uncharted"):
            idx = int(k.split(":")[1]); col = palette(np.array([0.12 + 0.24 * idx]))[0].astype(np.float32)
            cxy = ctr + P[k].mean(0)[:2] * sc * np.array([1, -1])
            n = 180; xs = np.linspace(ctr[0], cxy[0], n); ys = np.linspace(ctr[1], cxy[1], n)
            _splat(buf, xs, ys, np.broadcast_to(col, (n, 3)).copy(), 0.05)
            _splat(buf, xy[:, 0], xy[:, 1], np.broadcast_to(col, (len(xy), 3)).copy(), 0.9)
        else:
            col = (np.array([0.55, 0.72, 0.95], np.float32) if k == "baseline"
                   else np.array([0.82, 0.66, 0.98], np.float32))
            _splat(buf, xy[:, 0], xy[:, 1], np.broadcast_to(col, (len(xy), 3)).copy(), 0.28)
    img = bloom(buf)
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.text((16, 12), "STYLE 4 — Manifold + Rays  (shared embedding; core = baseline+named affective cloud,",
           font=_font(18), fill=(220, 220, 235))
    d.text((16, 36), "each coloured ray = one uncharted state flying off-manifold in its OWN direction)",
           font=_font(18), fill=(220, 220, 235))
    im.save(OUT / "05_manifold_rays.png")
    print("wrote", OUT / "05_manifold_rays.png")


# ── contact sheet ─────────────────────────────────────────────────────────────
def _font(sz):
    for p in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def sheet(cells, title, fname, cols=3):
    pad, lab = 14, 26
    rows = math.ceil(len(cells) / cols)
    W = cols * S + (cols + 1) * pad
    Hh = 44 + rows * (S + lab) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, Hh), (8, 6, 16))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 12), title, font=_font(22), fill=(220, 220, 235))
    for i, (img01, caption) in enumerate(cells):
        r, c = divmod(i, cols)
        x = pad + c * (S + pad); ytop = 44 + pad + r * (S + lab + pad)
        canvas.paste(Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8)), (x, ytop + lab))
        d.text((x + 4, ytop + 4), caption, font=_font(15), fill=(180, 200, 220))
    canvas.save(OUT / fname)
    print("wrote", OUT / fname)


ORDER = ["baseline", "named:awe", "uncharted:0", "uncharted:1", "uncharted:2", "uncharted:3"]


def main():
    blob = torch.load(DATA / "uncharted_traj.pt", weights_only=False)
    runs = blob["runs"]
    # geometry for prompt-0 of each condition; eff-dim across set -> hue scale
    geo, eff = {}, {}
    for key in ORDER:
        g = geometry(runs[f"{key}|p0"]["traj"].float().numpy())
        geo[key] = g; eff[key] = g["effdim"]
    lo, hi = min(eff.values()), max(eff.values())
    hue = {k: (eff[k] - lo) / (hi - lo + 1e-9) for k in ORDER}
    print("eff-dim:", {k: round(v, 2) for k, v in eff.items()})

    man, cym, neb = [], [], []
    for key in ORDER:
        g = geo[key]; cap = f"{key}   eff-dim={g['effdim']:.2f}"
        man.append((bloom(mandala(g, hue[key])), cap))
        cym.append((bloom(cymatic(g, hue[key])), cap))
        neb.append((bloom(nebula(g)), cap))
    shared_map(runs)

    # signature mandala: spectrum -> shape, direction -> fingerprint + hue
    sig, _, basis, gm, mask = shared_embedding(runs)
    sigsheet = []
    for key in ORDER:
        g = geo[key]
        sigsheet.append((bloom(mandala_sig(g, sig[key])), f"{key}   eff-dim={g['effdim']:.2f}"))
    sheet(sigsheet, "STYLE 6 — Signature Mandala  (eigenspectrum -> shape; DIRECTION -> overtones + hue + chirality)", "06_mandala_sig.png")

    # reproducibility of the signature mandala: same direction, two prompts
    sig2 = []
    for key in ["uncharted:0", "uncharted:1", "uncharted:2", "uncharted:3"]:
        for pi in (0, 1):
            traj = runs[f"{key}|p{pi}"]["traj"].float().numpy()
            g = geometry(traj)
            sig_p = (traj[:, mask].mean(0) - gm) @ basis.T   # this prompt's own signature
            sig2.append((bloom(mandala_sig(g, sig_p)), f"{key}  prompt {pi}"))
    sheet(sig2, "STYLE 6 REPRO — same direction across two prompts. Distinct between rows, matched within = real fingerprint.", "07_mandala_sig_repro.png", cols=2)

    sheet(man, "STYLE 1 — Spectral Mandala  (eigenspectrum -> harmonic lobes; complexity = eff-dim)", "01_mandala.png")
    sheet(cym, "STYLE 2 — Cymatic Field  (spectrum synthesised as a standing-wave interference pattern)", "02_cymatic.png")
    sheet(neb, "STYLE 3 — Nebula  (literal 3D PCA cloud as luminous volume; tight glow = real attractor)", "03_nebula.png")

    # reproducibility demo (mandala): same direction, two different prompts
    repro = []
    for key in ["uncharted:0", "uncharted:1", "uncharted:2"]:
        for pi in (0, 1):
            g = geometry(runs[f"{key}|p{pi}"]["traj"].float().numpy())
            h = (g["effdim"] - lo) / (hi - lo + 1e-9)
            repro.append((bloom(mandala(g, np.clip(h, 0, 1))), f"{key}  prompt {pi}"))
    sheet(repro, "REPRODUCIBILITY — same direction across two prompts (mandala). Similar = the structure is real, not prompt-noise.",
          "04_repro_mandala.png", cols=2)


if __name__ == "__main__":
    main()
