import {
  BackSide,
  CircleGeometry,
  DirectionalLight,
  Fog,
  HalfFloatType,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  NeutralToneMapping,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  Vector2,
  Vector3,
  VSMShadowMap,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { PALETTE } from "./palette";

const SHADOW_HALF_SPAN = 72;
const VECTOR = new Vector3();

/** Renderer, studio light, ground and the post stack. Knows nothing about traffic. */
export class Stage {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(30, 1, 1, 360);
  readonly renderer: WebGLRenderer;
  private readonly composer: EffectComposer;
  private readonly ao: GTAOPass;
  private paneWidth = 0;
  private paneHeight = 0;

  constructor(readonly canvas: HTMLCanvasElement) {
    const renderer = new WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.toneMapping = NeutralToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = VSMShadowMap;
    this.renderer = renderer;

    this.scene.background = PALETTE.background;
    this.scene.fog = new Fog(PALETTE.background, 60, 230);

    const environment = new PMREMGenerator(renderer);
    this.scene.environment = environment.fromScene(new RoomEnvironment(), 0.06).texture;
    this.scene.environmentIntensity = 0.62;
    environment.dispose();

    const key = new DirectionalLight("#fff4e6", 1.75);
    key.position.set(-26, 62, 24);
    key.target.position.set(0, 0, -26);
    key.castShadow = true;
    key.shadow.mapSize.set(4096, 4096);
    key.shadow.camera.left = -SHADOW_HALF_SPAN;
    key.shadow.camera.right = SHADOW_HALF_SPAN;
    key.shadow.camera.top = SHADOW_HALF_SPAN;
    key.shadow.camera.bottom = -SHADOW_HALF_SPAN;
    key.shadow.camera.near = 10;
    key.shadow.camera.far = 170;
    key.shadow.radius = 5;
    key.shadow.blurSamples = 16;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.03;
    this.scene.add(key, key.target);

    // A real surface behind everything: the AO pass darkens the bare background otherwise.
    const sky = new Mesh(
      new SphereGeometry(340, 32, 16),
      new MeshBasicMaterial({ color: PALETTE.background, side: BackSide, fog: false }),
    );
    this.scene.add(sky);

    const ground = new Mesh(
      new CircleGeometry(300, 96),
      new MeshStandardMaterial({ color: PALETTE.ground, roughness: 1, metalness: 0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    const target = new WebGLRenderTarget(1, 1, { type: HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.ao = new GTAOPass(this.scene, this.camera, 1, 1);
    this.ao.blendIntensity = 0.9;
    this.ao.updateGtaoMaterial({
      radius: 1.1,
      distanceExponent: 1.4,
      thickness: 1,
      scale: 1.15,
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
    this.composer.addPass(this.ao);
    this.composer.addPass(new OutputPass());

    this.fitToPane();
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
    this.composer.setPixelRatio(pixelRatio);
    this.composer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /** Screen position of a world point, in CSS pixels of the canvas. */
  project(x: number, y: number, z: number, out: Vector2): boolean {
    const ndc = VECTOR.set(x, y, z).project(this.camera);
    if (ndc.z > 1 || ndc.z < -1) return false;
    out.set(((ndc.x + 1) / 2) * this.paneWidth, ((1 - ndc.y) / 2) * this.paneHeight);
    return true;
  }

  render(): void {
    this.fitToPane();
    this.composer.render();
  }
}
