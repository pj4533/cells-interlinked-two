"use client";

import type { ComponentType } from "react";

import { BarsField } from "./BarsField";
import { CloudFlow } from "./CloudFlow";
import type { PlaybackVizProps } from "./types";

export type PlaybackVizKind = "bars" | "clouds";

/** Visualization registry. To add a new one, drop the component into
 *  this folder and add an entry here. */
export const PLAYBACK_VIZ_REGISTRY: Record<
  PlaybackVizKind,
  ComponentType<PlaybackVizProps>
> = {
  bars: BarsField,
  clouds: CloudFlow,
};

/** Active visualization. Change this constant to swap. Eventually
 *  this could become a runtime toggle in the UI, but for iteration
 *  a constant is the cheapest way to compare side-by-side. */
export const ACTIVE_PLAYBACK_VIZ: PlaybackVizKind = "clouds";

export { attachAudio, getAnalyser } from "./audio-graph";
export type { PlaybackVizProps } from "./types";
