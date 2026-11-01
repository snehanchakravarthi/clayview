import { Color } from "three";

/** One place for the scene's colours. Warm greys, with terracotta reserved for ego. */
export const PALETTE = {
  background: new Color("#f8f1e4"),
  ground: new Color("#dfd9cd"),
  road: new Color("#d3cdc1"),
  marking: new Color("#f5f2ec"),
  ring: new Color("#8f887b"),
  foam: new Color("#f3efe8"),
  foamGlass: new Color("#b9b2a6"),
  foamTyre: new Color("#8b8478"),
  clay: new Color("#c4704a"),
  clayGlass: new Color("#7e4a35"),
  clayTyre: new Color("#5e4639"),
  lead: new Color("#e9c9a8"),
} as const;
