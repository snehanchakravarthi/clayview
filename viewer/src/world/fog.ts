import { ShaderChunk } from "three";

/** Fog is measured from a point ahead of ego, on the ground, not from the camera. */
export const FOG_FOCUS_Z = -34;

let installed = false;

/**
 * Three's fog thickens with depth from the camera, which would wash out the top view from its own
 * height. Here it's a ring around the car instead: every view gets a clear island of street that
 * dissolves into the studio backdrop at the same distance, and tall things go no sooner than low
 * ones. A material can define FOG_SCALE to thin out earlier.
 */
export function installRadialFog(): void {
  if (installed) return;
  installed = true;
  ShaderChunk.fog_pars_vertex = /* glsl */ `
    #ifdef USE_FOG
      varying vec2 vFogPlane;
    #endif
  `;
  ShaderChunk.fog_vertex = /* glsl */ `
    #ifdef USE_FOG
      vFogPlane = (transpose(mat3(viewMatrix)) * mvPosition.xyz + cameraPosition).xz;
    #endif
  `;
  ShaderChunk.fog_pars_fragment = /* glsl */ `
    #ifdef USE_FOG
      uniform vec3 fogColor;
      varying vec2 vFogPlane;
      uniform float fogNear;
      uniform float fogFar;
    #endif
  `;
  ShaderChunk.fog_fragment = /* glsl */ `
    #ifdef USE_FOG
      float fogRange = length(vFogPlane - vec2(0.0, ${FOG_FOCUS_Z.toFixed(1)}));
      #ifdef FOG_SCALE
        fogRange *= float(FOG_SCALE);
      #endif
      float fogFactor = smoothstep(fogNear, fogFar, fogRange);
      gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, fogFactor);
    #endif
  `;
}
