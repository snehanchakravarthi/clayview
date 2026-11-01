import { type Material, type PerspectiveCamera, Vector2, Vector3 } from "three";

/** How far around ego, in metres at ego's distance, a line of sight is kept open. */
const SIGHT_RADIUS_M = 6.5;
const CHASE_CLOSE = 34;
const CHASE_FAR = 60;

const uniforms = {
  /** Ego on screen in device pixels, and the radius of the opening. */
  cutawayEgo: { value: new Vector3() },
  /** Distance from camera to ego. */
  cutawayDepth: { value: 0 },
  /** Distance range over which things in front of a close camera dissolve. */
  cutawayNear: { value: new Vector2() },
};

const POINT = new Vector3();

/**
 * Lets tall things stand on the street without hiding it. Anything between a close camera and the
 * car dissolves, and from further out a soft hole stays open around the car. It works by dithered
 * discard, so there's no sorting, and the AO reads the same depth, so nothing leaves a ghost.
 * Shadows don't go through this, so a dissolved tree still shades the road.
 */
export function aimCutaway(camera: PerspectiveCamera, width: number, height: number): void {
  POINT.set(0, 0.9, 0);
  const distance = POINT.distanceTo(camera.position);
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  POINT.project(camera);
  const pixelsPerMetre =
    height / (2 * Math.tan((camera.fov * Math.PI) / 360) * Math.max(distance, 1));
  uniforms.cutawayEgo.value.set(
    ((POINT.x + 1) / 2) * width,
    ((POINT.y + 1) / 2) * height,
    SIGHT_RADIUS_M * pixelsPerMetre,
  );
  uniforms.cutawayDepth.value = distance;
  const t = Math.min(Math.max((distance - CHASE_CLOSE) / (CHASE_FAR - CHASE_CLOSE), 0), 1);
  const close = 1 - t * t * (3 - 2 * t);
  uniforms.cutawayNear.value.set(distance * 0.7 * close, distance * 0.9 * close);
}

/**
 * Make a standard material give way to the line of sight. With `whole`, every instance dissolves as
 * one piece, judged at its middle: a crown sliced by distance would show that it's hollow.
 */
export function withCutaway<T extends Material>(material: T, whole = false): T {
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    if (whole) {
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying float vCutRange;")
        .replace(
          "#include <project_vertex>",
          /* glsl */ `#include <project_vertex>
          vCutRange = length((modelViewMatrix * instanceMatrix * vec4(0.0, 0.55, 0.0, 1.0)).xyz);`,
        );
    }
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        /* glsl */ `#include <common>
        uniform vec3 cutawayEgo;
        uniform float cutawayDepth;
        uniform vec2 cutawayNear;
        ${whole ? "varying float vCutRange;" : ""}`,
      )
      .replace(
        "#include <clipping_planes_fragment>",
        /* glsl */ `#include <clipping_planes_fragment>
        {
          float range = ${whole ? "vCutRange" : "length(vViewPosition)"};
          float keep = cutawayNear.y > 0.0 ? smoothstep(cutawayNear.x, cutawayNear.y, range) : 1.0;
          float offset = length(gl_FragCoord.xy - cutawayEgo.xy) / cutawayEgo.z;
          float infront = smoothstep(cutawayDepth - 1.0, cutawayDepth - 7.0, range);
          keep = min(keep, 1.0 - infront * (1.0 - smoothstep(0.65, 1.25, offset)));
          float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
          if (noise >= keep) discard;
        }`,
      );
  };
  material.customProgramCacheKey = (): string => (whole ? "cutaway-whole" : "cutaway");
  return material;
}
