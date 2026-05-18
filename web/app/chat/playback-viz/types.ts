/** Common contract for chat-mode playback visualizations.
 *
 * The audio analyser is a singleton (`audio-graph.ts`); visualizations
 * pull live frequency / waveform data from it themselves via
 * `getAnalyser()`. The component props only carry the channel-tinted
 * colors, which are channel-α (raw, amber) or channel-β (ablated,
 * cyan) depending on which side is currently playing.
 *
 * To add a new visualization:
 *   1. Drop a new <Name>.tsx in this directory implementing the
 *      `(props: PlaybackVizProps) => JSX.Element` contract.
 *   2. Register it in `index.ts`.
 *   3. Set `ACTIVE_PLAYBACK_VIZ` to its key to switch.
 */

export interface PlaybackVizProps {
  /** CSS color string for the side currently speaking, e.g.
   * "rgba(232,195,130,0.95)" for raw or "rgba(94,229,229,0.95)" for
   * ablated. Use for primary fills / strokes. */
  accent: string;
  /** Same color but as bare "r,g,b" components so the visualization
   * can compose rgba()s at arbitrary alpha values for layered fills
   * and glows. */
  accentRgb: string;
}
