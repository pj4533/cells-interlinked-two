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

const CYAN = new THREE.Color("#5ee5e5"); // baseline / raw
const AMBER = new THREE.Color("#e8c382"); // ablated / off-manifold
const HOT = new THREE.Color("#ff7a3c"); // tokens the ablation displaces most
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
    const hue = alphaMax > 0 ? Math.min(1, a / alphaMax) : 0; // 0 raw → 1 ablated

    // Morph raw→ablated in projection space, then normalize to world scale.
    ablatedCoordsInto(work, geometry, a);
    for (let i = 0; i < N; i++) {
      pos[i * 3 + 0] = work[i * 3 + 0] * inv * WORLD;
      pos[i * 3 + 1] = work[i * 3 + 1] * inv * WORLD;
      pos[i * 3 + 2] = work[i * 3 + 2] * inv * WORLD;

      // Color: cyan→amber by global α, pushed toward HOT for tokens the
      // ablation displaces most. Reveal gates brightness (unfold).
      tmp.copy(CYAN).lerp(AMBER, hue);
      const heat = dispMax[i] * hue;
      tmp.lerp(HOT, heat * 0.6);
      const shown = i < revealCount ? 1 : i < revealCount + 1 ? reveal.current % 1 : 0;
      const lead = i > revealCount - 6 && i < revealCount ? 1.6 : 1; // bright leading edge
      col[i * 3 + 0] = tmp.r * shown * lead;
      col[i * 3 + 1] = tmp.g * shown * lead;
      col[i * 3 + 2] = tmp.b * shown * lead;
    }
    (bufGeom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (bufGeom.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    (lineGeom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (lineGeom.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    // Only draw the path up to the revealed frontier.
    lineGeom.setDrawRange(0, Math.max(0, Math.floor(revealCount)));
  });

  return (
    <group>
      <points ref={pointsRef} geometry={bufGeom}>
        <pointsMaterial
          map={glow}
          size={0.42}
          sizeAttenuation
          vertexColors
          transparent
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
      color: new THREE.Color("#8a7349"),
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
