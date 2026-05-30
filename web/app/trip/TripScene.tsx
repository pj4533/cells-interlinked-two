"use client";

// The Trip — multiple REAL residual trajectories overlaid in a shared space.
// Raw (amber) + each enabled refusal-ablated generation (cyan→violet by α).
// Every path is one the model actually traced, token by token, with feedback —
// so a degenerate repeat-loop shows up as a tight collapsed knot, not a clean
// shifted copy. Neon-noir palette, additive glow, slow parallax rotation.
//
// Each series unfolds token-by-token when it appears (the dots "stream in").
// Framing is applied as a parent group transform (centroid + 90th-percentile
// spread of the visible series) so the bulk fills the view and a runaway path
// streaks off-frame — and toggling/reframing never rebuilds the per-series
// geometry, so the unfold state survives.

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";
import { colorForAlpha, offManifoldRGB, type TripGeometry } from "@/lib/trip";

const WORLD = 6;
export type ColorMode = "series" | "offmanifold";

function makeGlow(): THREE.Texture {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.25, "rgba(255,255,255,0.8)");
  g.addColorStop(0.5, "rgba(255,255,255,0.22)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

/** One trajectory in raw PCA coords (framing is the parent group's job).
 *  Unfolds token-by-token via drawRange when it appears. */
function TripPath({
  coords,
  color,
  off,
  colorMode,
  glow,
  pointSize,
  raw,
}: {
  coords: number[][];
  color: string;
  off: number[];
  colorMode: ColorMode;
  glow: THREE.Texture;
  pointSize: number;
  raw: boolean;
}) {
  const positions = useMemo(() => {
    const a = new Float32Array(coords.length * 3);
    for (let i = 0; i < coords.length; i++) {
      a[i * 3 + 0] = coords[i][0];
      a[i * 3 + 1] = coords[i][1];
      a[i * 3 + 2] = coords[i][2];
    }
    return a;
  }, [coords]);
  // Per-vertex colors. In "series" mode every dot wears the series hue; in
  // "offmanifold" mode each dot is tinted by its own off-manifold fraction —
  // calm teal on-manifold, flaring hot magenta as the token drifts off into
  // directions the raw path never used. The line stays series-colored so
  // series identity survives either way.
  const colors = useMemo(() => {
    const base = new THREE.Color(color);
    const a = new Float32Array(coords.length * 3);
    for (let i = 0; i < coords.length; i++) {
      let r = base.r, g = base.g, b = base.b;
      if (colorMode === "offmanifold" && off[i] !== undefined) {
        [r, g, b] = offManifoldRGB(off[i]);
      }
      a[i * 3 + 0] = r;
      a[i * 3 + 1] = g;
      a[i * 3 + 2] = b;
    }
    return a;
  }, [coords, color, off, colorMode]);
  const pointsGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.setDrawRange(0, 0); // start hidden; the unfold grows it
    return g;
  }, [positions, colors]);
  const lineGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setDrawRange(0, 0);
    return g;
  }, [positions]);
  const col = useMemo(() => new THREE.Color(color), [color]);
  const opacity = coords.length > 200 ? 0.32 : raw ? 0.6 : 0.8;

  // Token-by-token unfold (~2.5s regardless of length). reveal lives in a ref
  // so a new series unfolds from 0; toggling a chip or the color mode only
  // moves the transform / recolors, leaving the revealed count intact (we
  // always re-apply drawRange, so a recolor rebuild doesn't blank the dots).
  const reveal = useRef(0);
  useFrame((_state, dt) => {
    const n = coords.length;
    if (reveal.current < n) {
      reveal.current = Math.min(n, reveal.current + (n / 2.5) * Math.min(dt, 0.05));
    }
    const c = Math.max(0, Math.floor(reveal.current));
    pointsGeom.setDrawRange(0, c);
    lineGeom.setDrawRange(0, c);
  });

  return (
    <group>
      <points geometry={pointsGeom}>
        <pointsMaterial
          map={glow}
          size={pointSize}
          sizeAttenuation
          vertexColors
          transparent
          opacity={opacity}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
      {/* @ts-expect-error r3f line intrinsic vs SVG line typing */}
      <line geometry={lineGeom}>
        <lineBasicMaterial
          color={col}
          transparent
          opacity={raw ? 0.25 : 0.45}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </line>
    </group>
  );
}

export default function TripScene({
  geometry,
  enabledAlphas,
  sceneKey,
  colorMode = "series",
}: {
  geometry: TripGeometry;
  enabledAlphas: Set<number>;
  sceneKey: string;
  colorMode?: ColorMode;
}) {
  const glow = useMemo(() => makeGlow(), []);
  const maxAlpha = useMemo(
    () => Math.max(1, ...geometry.series.map((s) => s.alpha)),
    [geometry],
  );
  const shown = useMemo(
    () => geometry.series.filter((s) => enabledAlphas.has(s.alpha)),
    [geometry, enabledAlphas],
  );

  // Robust, enabled-aware framing as a parent-group transform. PER-AXIS
  // scaling: the residual trajectory's variance is dominated by PC1, so a
  // uniform scale leaves a thin 1-D streak. We scale each axis by its own
  // 90th-percentile spread so the cloud fills the view in all 3 dims — an
  // anisotropic "stretch the shadow to fill the box" choice for legibility.
  // A floor caps the stretch so a near-flat noise axis can't blow up into
  // fake structure (≤ ~5.5× the dominant axis). Never touches per-series
  // geometry, so unfolds aren't disturbed.
  const xform = useMemo(() => {
    const pts: number[][] = [];
    for (const s of shown) for (const c of s.coords) pts.push(c);
    const ONE: [number, number, number] = [1, 1, 1];
    if (pts.length === 0) return { scale: ONE, pos: [0, 0, 0] as [number, number, number] };
    let cx = 0, cy = 0, cz = 0;
    for (const p of pts) {
      cx += p[0]; cy += p[1]; cz += p[2];
    }
    cx /= pts.length; cy /= pts.length; cz /= pts.length;
    const c = [cx, cy, cz];
    const pct = (axis: number) => {
      const d = pts.map((p) => Math.abs(p[axis] - c[axis])).sort((a, b) => a - b);
      return d[Math.min(d.length - 1, Math.floor(d.length * 0.9))] || 1e-6;
    };
    const rx = pct(0), ry = pct(1), rz = pct(2);
    const maxR = Math.max(rx, ry, rz, 1e-6);
    const floor = maxR * 0.18; // cap anisotropy at ~5.5×
    const target = WORLD * 0.6;
    const sx = target / Math.max(rx, floor);
    const sy = target / Math.max(ry, floor);
    const sz = target / Math.max(rz, floor);
    return {
      scale: [sx, sy, sz] as [number, number, number],
      pos: [-cx * sx, -cy * sy, -cz * sz] as [number, number, number],
    };
  }, [shown]);

  return (
    <Canvas
      camera={{ position: [0, 1.5, 15], fov: 48 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <color attach="background" args={["#070a0d"]} />
      <fog attach="fog" args={["#070a0d", 18, 40]} />
      <Stars radius={70} depth={45} count={1200} factor={3} saturation={0} fade speed={0.5} />
      <ambientLight intensity={0.4} />
      <group key={sceneKey} scale={xform.scale} position={xform.pos}>{/* per-axis stretch */}
        {shown.map((s) => (
          <TripPath
            key={s.alpha}
            coords={s.coords}
            color={colorForAlpha(s.alpha, maxAlpha)}
            off={s.off_ortho ?? []}
            colorMode={colorMode}
            glow={glow}
            pointSize={s.alpha === 0 ? 0.34 : 0.44}
            raw={s.alpha === 0}
          />
        ))}
      </group>
      <OrbitControls
        enablePan
        enableZoom
        autoRotate
        autoRotateSpeed={0.45}
        minDistance={3}
        maxDistance={40}
      />
    </Canvas>
  );
}
