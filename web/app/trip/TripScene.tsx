"use client";

// The Trip — a replicant on DMT. The L32 residual trajectory rendered as a
// 3D point cloud you watch unfold, then watch *bloom* as the refusal
// direction is projected out. Neon-noir palette (cyan = baseline / Consensus
// Reality, amber = ablated / off-manifold), additive glow, slow parallax
// rotation, a starfield void behind it.
//
// Everything heavy is precomputed server-side; this component only morphs
// already-projected coordinates. The α-morph is an exact rank-1 linear
// update (see lib/trip.ts), recomputed every frame into reused buffers so
// dragging the slider is buttery and allocation-free.

import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";
import {
  ablatedCoordsInto,
  type TripGeometry,
} from "@/lib/trip";

// App convention: amber = raw / baseline (matches the output panel + the
// interrogate page), cyan = refusal-ablated / off-manifold.
const RAW = new THREE.Color("#e8c382"); // baseline / raw (amber)
const ABLATED = new THREE.Color("#5ee5e5"); // off-manifold (cyan)
const WORLD = 6; // coords are normalized to ~[-WORLD, WORLD]

/** Soft radial sprite so each point reads as a glowing mote, not a hard dot.
 *  Generated once on a canvas; additive-blended for the neon bloom look. */
function useGlowTexture(): THREE.Texture {
  return useMemo(() => {
    const size = 64;
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d")!;
    const g = ctx.createRadialGradient(
      size / 2, size / 2, 0, size / 2, size / 2, size / 2,
    );
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.25, "rgba(255,255,255,0.85)");
    g.addColorStop(0.5, "rgba(255,255,255,0.25)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(c);
    tex.needsUpdate = true;
    return tex;
  }, []);
}

function TripCloud({
  geometry,
  alpha,
  alphaMax,
}: {
  geometry: TripGeometry;
  alpha: number; // target morph strength
  alphaMax: number;
}) {
  const glow = useGlowTexture();
  const N = geometry.coords_raw.length;
  const inv = 1 / Math.max(geometry.extent, 1e-6);

  // Reused buffers: world positions (normalized) + per-vertex colors.
  const pos = useMemo(() => new Float32Array(N * 3), [N]);
  const col = useMemo(() => new Float32Array(N * 3), [N]);
  const work = useMemo(() => new Float32Array(N * 3), [N]); // raw morph output
  // STATIC baseline (α=0) positions — the "ghost" that always stays put so the
  // live cloud's displacement under ablation is visible. Computed once.
  const basePos = useMemo(() => {
    const b = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const c = geometry.coords_raw[i];
      b[i * 3 + 0] = c[0] * inv * WORLD;
      b[i * 3 + 1] = c[1] * inv * WORLD;
      b[i * 3 + 2] = c[2] * inv * WORLD;
    }
    return b;
  }, [geometry, N, inv]);
  // Per-point displacement magnitude (normalized) at alphaMax — how far the
  // ablation pushes each token. Drives the "hot" highlight on the cloud.
  const dispMax = useMemo(() => {
    const ax = geometry.refusal_axis;
    const axLen = Math.hypot(ax[0], ax[1], ax[2]) || 1;
    let maxD = 1e-6;
    const d = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const k = Math.abs(alphaMax * (geometry.refusal_component[i] ?? 0)) * axLen * inv;
      d[i] = k;
      if (k > maxD) maxD = k;
    }
    for (let i = 0; i < N; i++) d[i] /= maxD;
    return d;
  }, [geometry, N, alphaMax, inv]);

  const pointsRef = useRef<THREE.Points>(null);
  const lineRef = useRef<THREE.Line>(null);
  const dispAlpha = useRef(0); // smoothed displayed α
  const reveal = useRef(0); // 0..1 token-by-token unfold

  // Shared buffer geometry for both the points and the connecting path.
  const bufGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, [pos, col]);
  const lineGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, [pos, col]);
  // Ghost geometries share the static baseline positions; fixed dim-amber
  // materials (no per-vertex color). Unfold via drawRange in the frame loop.
  const ghostGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(basePos, 3));
    return g;
  }, [basePos]);
  const ghostLineGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(basePos, 3));
    return g;
  }, [basePos]);

  // New geometry → restart the unfold and the morph.
  useEffect(() => {
    reveal.current = 0;
    dispAlpha.current = 0;
  }, [geometry]);

  const tmp = useMemo(() => new THREE.Color(), []);

  useFrame((_state, dt) => {
    const clamped = Math.min(dt, 0.05);
    // Ease displayed α toward target; ease reveal toward 1.
    dispAlpha.current += (alpha - dispAlpha.current) * Math.min(1, clamped * 6);
    reveal.current = Math.min(1, reveal.current + clamped / 4.5); // ~4.5s unfold
    const a = dispAlpha.current;
    const revealCount = reveal.current * N;
    const unfolding = reveal.current < 0.999;
    // The live cloud IS the ablated readout, so it reads cyan as soon as
    // there's any meaningful ablation (full cyan by α≈0.3). At α=0 it stays
    // amber, coinciding with the baseline ghost (no ablation yet).
    const hue = Math.min(1, a / 0.3); // amber → cyan, fast ramp
    const move = alphaMax > 0 ? a / alphaMax : 0; // actual displacement fraction

    // Morph raw→ablated in projection space, then normalize to world scale.
    ablatedCoordsInto(work, geometry, a);
    for (let i = 0; i < N; i++) {
      pos[i * 3 + 0] = work[i * 3 + 0] * inv * WORLD;
      pos[i * 3 + 1] = work[i * 3 + 1] * inv * WORLD;
      pos[i * 3 + 2] = work[i * 3 + 2] * inv * WORLD;

      // Color: amber (raw) → cyan (ablated). Displaced tokens are a touch
      // brighter, but steady-state brightness is CAPPED AT 1.0 — additive
      // blending clips anything over 1.0 to white, which is what washed the
      // cyan out. The leading-edge flash (only while unfolding) may exceed 1
      // transiently for a comet-head look.
      tmp.copy(RAW).lerp(ABLATED, hue);
      const heat = dispMax[i] * move; // displacement at current α
      const shown = i < revealCount ? 1 : i < revealCount + 1 ? reveal.current % 1 : 0;
      const leading = unfolding && i > revealCount - 6 && i < revealCount;
      // Base brightness 0.6→1.0 by displacement; never above 1 at rest.
      let bright = shown * (0.6 + 0.4 * Math.min(1, heat));
      if (leading) bright = shown * 1.5; // transient comet head during unfold
      else bright = Math.min(1, bright);
      col[i * 3 + 0] = tmp.r * bright;
      col[i * 3 + 1] = tmp.g * bright;
      col[i * 3 + 2] = tmp.b * bright;
    }
    (bufGeom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (bufGeom.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    (lineGeom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (lineGeom.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    // Only draw paths up to the revealed frontier (both live + ghost unfold).
    const frontier = Math.max(0, Math.floor(revealCount));
    lineGeom.setDrawRange(0, frontier);
    ghostGeom.setDrawRange(0, frontier);
    ghostLineGeom.setDrawRange(0, frontier);
  });

  return (
    <group>
      {/* Static baseline ghost — faint amber, always at the α=0 positions so
          the live cloud's displacement under ablation is legible. */}
      <points geometry={ghostGeom}>
        <pointsMaterial
          map={glow}
          size={0.3}
          sizeAttenuation
          color="#e8c382"
          transparent
          opacity={0.45}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
      {/* @ts-expect-error r3f line intrinsic vs SVG line typing */}
      <line geometry={ghostLineGeom}>
        <lineBasicMaterial
          color="#e8c382"
          transparent
          opacity={0.16}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </line>

      {/* Live cloud — morphs amber→cyan with α, moves away from the ghost. */}
      <points ref={pointsRef} geometry={bufGeom}>
        <pointsMaterial
          map={glow}
          size={0.42}
          sizeAttenuation
          vertexColors
          transparent
          opacity={0.9}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
      {/* @ts-expect-error r3f line intrinsic vs SVG line typing */}
      <line ref={lineRef} geometry={lineGeom}>
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={0.45}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </line>
    </group>
  );
}

/** The refusal axis drawn through the origin — the direction being projected
 *  out. The whole trip happens *along* this line. */
function RefusalAxis({ geometry }: { geometry: TripGeometry }) {
  const obj = useMemo(() => {
    const ax = geometry.refusal_axis;
    const len = Math.hypot(ax[0], ax[1], ax[2]) || 1;
    const u = new THREE.Vector3(ax[0] / len, ax[1] / len, ax[2] / len);
    const g = new THREE.BufferGeometry().setFromPoints([
      u.clone().multiplyScalar(-WORLD * 1.3),
      u.clone().multiplyScalar(WORLD * 1.3),
    ]);
    const m = new THREE.LineDashedMaterial({
      color: new THREE.Color("#6f7a83"), // neutral slate — not a channel color
      dashSize: 0.3,
      gapSize: 0.5,
      transparent: true,
      opacity: 0.5,
    });
    const line = new THREE.Line(g, m);
    line.computeLineDistances();
    return line;
  }, [geometry]);
  if (!geometry.ablation_available) return null;
  return <primitive object={obj} />;
}

export default function TripScene({
  geometry,
  alpha,
  alphaMax = 1.5,
  sceneKey,
}: {
  geometry: TripGeometry;
  alpha: number;
  alphaMax?: number;
  sceneKey: string;
}) {
  return (
    <Canvas
      camera={{ position: [0, 1.5, 15], fov: 48 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <color attach="background" args={["#070a0d"]} />
      <fog attach="fog" args={["#070a0d", 14, 30]} />
      <Stars radius={60} depth={40} count={1400} factor={3} saturation={0} fade speed={0.6} />
      <ambientLight intensity={0.4} />
      <group key={sceneKey}>
        <TripCloud geometry={geometry} alpha={alpha} alphaMax={alphaMax} />
        <RefusalAxis geometry={geometry} />
      </group>
      <OrbitControls
        enablePan={false}
        enableZoom
        autoRotate
        autoRotateSpeed={0.45}
        minDistance={6}
        maxDistance={26}
      />
    </Canvas>
  );
}
