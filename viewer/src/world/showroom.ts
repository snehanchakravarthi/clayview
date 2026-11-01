import { Group } from "three";
import { Actor, FOAM } from "./actors";

/** The empty state: every model on the turntable, so the first thing you see is the clay. */
export function buildShowroom(): Group {
  const group = new Group();
  const placements: [string, number, number, number][] = [
    ["bus", -11.6, -10.5, 0.42],
    ["truck", -4.8, -13.5, 0.14],
    ["suv", 2.6, -12, -0.28],
    ["sedan", -4.2, 2.6, 0.5],
    ["hatch", 5.8, -3.4, -0.5],
    ["motorcycle", 5.2, 3.2, -0.7],
    ["bicycle", 7.6, 4.8, -0.9],
    ["person", 3.9, 6.2, -0.5],
  ];
  for (const [model, x, z, yaw] of placements) {
    const actor = new Actor(model, FOAM);
    actor.setOpacity(1);
    actor.group.position.set(x, 0, z);
    actor.group.rotation.y = yaw;
    group.add(actor.group);
  }
  return group;
}
