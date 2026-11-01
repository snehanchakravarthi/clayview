import {
  BackSide,
  CircleGeometry,
  DirectionalLight,
  DepthTexture,
  Fog,
  HalfFloatType,
  MathUtils,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  NeutralToneMapping,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  UnsignedIntType,
  Vector2,
  Vector3,
  VSMShadowMap,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { aimCutaway } from "./cutaway";
import { FOG_FOCUS_Z, installRadialFog } from "./fog";
import { PALETTE } from "./palette";

const SHADOW_HALF_SPAN = 96;
const SHADOW_MAP_SIZE = 4096;
const SHADOW_CENTRE_AHEAD = 44;
const SUN_DISTANCE = 110;
const VERTICAL_FOV = 30;
const WIDEST_FRAMED_ASPECT = 1.6;
const FOG_NEAR = 104;
const FOG_FAR = 214;
const GROUND_Y = -0.03;
const VECTOR = new Vector3();

/** Where the sun sits in a shot's world, as a unit vector towards it in three.js axes. */
const SUN = new Vector3(-0.4, 0.86, 0.34).normalize();
/** Shadow map axes in that same world frame, so its texels can be locked to the ground. */
const WORLD_UP = new Vector3(0, 1, 0);
const SUN_RIGHT = new Vector3().crossVectors(WORLD_UP, SUN).normalize();
const SUN_UP = new Vector3().crossVectors(SUN, SUN_RIGHT).normalize();
const FOCUS = new Vector3();

/** Renderer, studio light, ground and the post stack. Knows nothing about traffic. */
export class Stage {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(VERTICAL_FOV, 1, 1, 420);
  readonly renderer: WebGLRenderer;
  /** Colour and depth of the scene. The AO reads this depth, so it never redraws the world. */
  private readonly sceneTarget: WebGLRenderTarget;
  private readonly aoTarget: WebGLRenderTarget;
  private readonly ao: GTAOPass;
  private readonly output: OutputPass;
  private readonly key: DirectionalLight;
  private paneWidth = 0;
  private paneHeight = 0;
  /** What the scene pass alone drew: the shadow pass and post stack aren't counted here. */
  readonly drawn = { calls: 0, triangles: 0 };

  constructor(readonly canvas: HTMLCanvasElement) {
    installRadialFog();
    const renderer = new WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.toneMapping = NeutralToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = VSMShadowMap;
    this.renderer = renderer;

    this.scene.background = PALETTE.background;
    this.scene.fog = new Fog(PALETTE.background, FOG_NEAR, FOG_FAR);

    const environment = new PMREMGenerator(renderer);
    this.scene.environment = environment.fromScene(new RoomEnvironment(), 0.06).texture;
    this.scene.environmentIntensity = 0.62;
    environment.dispose();

    const key = new DirectionalLight("#fff4e6", 1.75);
    key.castShadow = true;
    key.shadow.mapSize.set(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    key.shadow.camera.left = -SHADOW_HALF_SPAN;
    key.shadow.camera.right = SHADOW_HALF_SPAN;
    key.shadow.camera.top = SHADOW_HALF_SPAN;
    key.shadow.camera.bottom = -SHADOW_HALF_SPAN;
    key.shadow.camera.near = 8;
    key.shadow.camera.far = SUN_DISTANCE + 110;
    key.shadow.radius = 5;
    key.shadow.blurSamples = 16;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.03;
    this.key = key;
    this.scene.add(key, key.target);
    this.aimSun(0, 0, 0);

    // A real surface behind everything: the AO pass darkens the bare background otherwise.
    const sky = new Mesh(
      new SphereGeometry(340, 32, 16),
      new MeshBasicMaterial({ color: PALETTE.background, side: BackSide, fog: false }),
    );
    this.scene.add(sky);

    const ground = new Mesh(
      new CircleGeometry(380, 96),
      new MeshStandardMaterial({ color: PALETTE.ground, roughness: 1, metalness: 0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = GROUND_Y;
    ground.receiveShadow = true;
    this.scene.add(ground);

    const depth = new DepthTexture(1, 1, UnsignedIntType);
    this.sceneTarget = new WebGLRenderTarget(1, 1, {
      type: HalfFloatType,
      samples: 4,
      depthTexture: depth,
    });
    this.aoTarget = new WebGLRenderTarget(1, 1, { type: HalfFloatType, depthBuffer: false });
    this.ao = new GTAOPass(this.scene, this.camera, 1, 1, { depthTexture: depth });
    this.ao.blendIntensity = 0.9;
    this.ao.updateGtaoMaterial({
      radius: 1.1,
      distanceExponent: 1.4,
      thickness: 1,
      scale: 1.3,
      samples: 16,
      distanceFallOff: 1,
      screenSpaceRadius: false,
    });
    this.ao.updatePdMaterial({
      lumaPhi: 10,
      depthPhi: 2,
      normalPhi: 3,
      radius: 6,
      radiusExponent: 1.5,
      rings: 2,
      samples: 16,
    });
    this.fadeAmbientOcclusion(depth);
    this.output = new OutputPass();
    this.output.renderToScreen = true;

    this.fitToPane();
  }

  /**
   * AO lands on top of finished colour, fog included, so far geometry would keep dark outlines
   * after the fog had swallowed it, and the sky sphere would show its facets. It fades with the fog.
   */
  private fadeAmbientOcclusion(depth: DepthTexture): void {
    const material = this.ao.blendMaterial;
    material.uniforms.tDepth = { value: depth };
    material.uniforms.fadeNear = { value: FOG_NEAR * 0.7 };
    material.uniforms.fadeFar = { value: FOG_FAR * 0.8 };
    material.uniforms.projectionInverse = { value: new Matrix4() };
    material.uniforms.cameraWorld = { value: new Matrix4() };
    material.fragmentShader = /* glsl */ `
      uniform float intensity;
      uniform sampler2D tDiffuse;
      uniform sampler2D tDepth;
      uniform float fadeNear;
      uniform float fadeFar;
      uniform mat4 projectionInverse;
      uniform mat4 cameraWorld;
      varying vec2 vUv;
      void main() {
        float depth = texture2D(tDepth, vUv).x;
        vec4 view = projectionInverse * vec4(vec3(vUv, depth) * 2.0 - 1.0, 1.0);
        vec4 world = cameraWorld * vec4(view.xyz / view.w, 1.0);
        float range = length(world.xz - vec2(0.0, ${FOG_FOCUS_Z.toFixed(1)}));
        float strength = intensity * (1.0 - smoothstep(fadeNear, fadeFar, range));
        vec4 texel = texture2D(tDiffuse, vUv);
        gl_FragColor = vec4(mix(vec3(1.0), texel.rgb, strength), texel.a);
      }
    `;
    material.needsUpdate = true;
  }

  /** Match the pane. Checked every frame: panes change size with the layout, not only the window. */
  private fitToPane(): void {
    const parent = this.canvas.parentElement ?? this.canvas;
    const width = Math.max(parent.clientWidth, 1);
    const height = Math.max(parent.clientHeight, 1);
    if (width === this.paneWidth && height === this.paneHeight) return;
    this.paneWidth = width;
    this.paneHeight = height;
    const pixelRatio = Math.min(window.devicePixelRatio, 2);
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(width, height, false);
    const pixelWidth = Math.round(width * pixelRatio);
    const pixelHeight = Math.round(height * pixelRatio);
    this.sceneTarget.setSize(pixelWidth, pixelHeight);
    this.aoTarget.setSize(pixelWidth, pixelHeight);
    this.ao.setSize(pixelWidth, pixelHeight);
    this.output.setSize(pixelWidth, pixelHeight);
    const aspect = width / height;
    // Narrow panes keep the same sideways reach, or parked rows beside the road fall out of frame.
    const halfHeight = Math.tan(MathUtils.degToRad(VERTICAL_FOV / 2));
    const fit = (halfHeight * WIDEST_FRAMED_ASPECT) / aspect;
    this.camera.aspect = aspect;
    this.camera.fov = MathUtils.radToDeg(2 * Math.atan(Math.max(fit, halfHeight)));
    this.camera.updateProjectionMatrix();
  }

  /** Screen position of a world point, in CSS pixels of the canvas. */
  project(x: number, y: number, z: number, out: Vector2): boolean {
    const ndc = VECTOR.set(x, y, z).project(this.camera);
    if (ndc.z > 1 || ndc.z < -1) return false;
    out.set(((ndc.x + 1) / 2) * this.paneWidth, ((1 - ndc.y) / 2) * this.paneHeight);
    return true;
  }

  /**
   * Put the sun where it belongs for ego's pose, so shadows stay fixed to the ground as the car
   * turns. The shadow map follows the car in whole texels: anything finer makes edges crawl.
   */
  aimSun(x: number, y: number, yaw: number): void {
    const sin = Math.sin(yaw);
    const cos = Math.cos(yaw);
    // The middle of the shadow map in the shot's world, in three.js axes.
    FOCUS.set(x + sin * SHADOW_CENTRE_AHEAD, 0, -(y + cos * SHADOW_CENTRE_AHEAD));
    const texel = (SHADOW_HALF_SPAN * 2) / SHADOW_MAP_SIZE;
    const right = FOCUS.dot(SUN_RIGHT);
    const up = FOCUS.dot(SUN_UP);
    FOCUS.addScaledVector(SUN_RIGHT, Math.round(right / texel) * texel - right);
    FOCUS.addScaledVector(SUN_UP, Math.round(up / texel) * texel - up);

    // World to ego: the same rigid move the street makes.
    const dx = FOCUS.x - x;
    const dz = FOCUS.z + y;
    this.key.target.position.set(dx * cos + dz * sin, FOCUS.y, -dx * sin + dz * cos);
    VECTOR.copy(SUN).applyAxisAngle(WORLD_UP, yaw);
    this.key.position.copy(this.key.target.position).addScaledVector(VECTOR, SUN_DISTANCE);
  }

  render(): void {
    this.fitToPane();
    const { renderer, scene, camera, sceneTarget, aoTarget } = this;
    aimCutaway(camera, sceneTarget.width, sceneTarget.height);
    renderer.setRenderTarget(sceneTarget);
    renderer.clear();
    renderer.render(scene, camera);
    this.drawn.calls = renderer.info.render.calls;
    this.drawn.triangles = renderer.info.render.triangles;
    const blend = this.ao.blendMaterial.uniforms;
    (blend.projectionInverse!.value as Matrix4).copy(camera.projectionMatrixInverse);
    (blend.cameraWorld!.value as Matrix4).copy(camera.matrixWorld);
    this.ao.render(renderer, aoTarget, sceneTarget, 0, false);
    this.output.render(renderer, aoTarget, aoTarget, 0, false);
  }
}
