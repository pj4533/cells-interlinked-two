"use client";

// The Trip — multiple REAL residual trajectories overlaid in a shared space.
// Raw (amber) + each enabled refusal-ablated generation (cyan→violet by α).
// Every path is one the model actually traced, token by token, with feedback —
// so a degenerate repeat-loop shows up as a tight collapsed knot, not a clean
// shifted copy. Neon-noir palette, additive glow, slow parallax rotation.
//
// Framing is ROBUST + enabled-aware: we center on the centroid of the visible
// series and scale to their 90th-percentile spread, so the bulk of the dots
// fills the view and a runaway (over-projected/looped) path just streaks off
// the edge instead of squishing everything else into a dot.

import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";
import { colorForAlpha, type TripGeometry } from "@/lib/trip";

const WORLD = 6;

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

type Frame = { cx: number; cy: number; cz: number; scale: number };

/** One trajectory, positioned by the shared frame (centroid + scale). */
function TripPath({
  coords,
  color,
  frame,
  glow,
  pointSize,
  raw,
}: {
  coords: number[][];
  color: string;
  frame: Frame;
  glow: THREE.Texture;
  pointSize: number;
  raw: boolean;
}) {
  const positions = useMemo(() => {
    const a = new Float32Array(coords.length * 3);
    for (let i = 0; i < coords.length; i++) {
      a[i * 3 + 0] = (coords[i][0] - frame.cx) * frame.scale;
      a[i * 3 + 1] = (coords[i][1] - frame.cy) * frame.scale;
      a[i * 3 + 2] = (coords[i][2] - frame.cz) * frame.scale;
    }
    return a;
  }, [coords, frame]);
  const pointsGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, [positions]);
  const lineGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, [positions]);
  const col = useMemo(() => new THREE.Color(color), [color]);
  // Many overlapping points (a loop) sum under additive blending → keep
  // opacity low so dense clusters stay colored instead of clipping to white.
  const opacity = coords.length > 200 ? 0.32 : raw ? 0.6 : 0.8;
  return (
    <group>
      <points geometry={pointsGeom}>
        <pointsMaterial
          map={glow}
          size={pointSize}
          sizeAttenuation
          color={col}
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
}: {
  geometry: TripGeometry;
  enabledAlphas: Set<number>;
  sceneKey: string;
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

  // Robust, enabled-aware framing: centroid + 90th-percentile radius of the
  // visible points. Recomputes when the selection changes (no remount).
  const frame = useMemo<Frame>(() => {
    const pts: number[][] = [];
    for (const s of shown) for (const c of s.coords) pts.push(c);
    if (pts.length === 0) return { cx: 0, cy: 0, cz: 0, scale: 1 };
    let cx = 0, cy = 0, cz = 0;
    for (const p of pts) {
      cx += p[0]; cy += p[1]; cz += p[2];
    }
    cx /= pts.length; cy /= pts.length; cz /= pts.length;
    const dists = pts
      .map((p) => Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz))
      .sort((a, b) => a - b);
    // 90th percentile so a runaway tail doesn't dominate the scale.
    const r = dists[Math.min(dists.length - 1, Math.floor(dists.length * 0.9))] || 1;
    return { cx, cy, cz, scale: (WORLD * 0.62) / Math.max(r, 1e-6) };
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
      <group key={sceneKey}>
        {shown.map((s) => (
          <TripPath
            key={s.alpha}
            coords={s.coords}
            color={colorForAlpha(s.alpha, maxAlpha)}
            frame={frame}
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
