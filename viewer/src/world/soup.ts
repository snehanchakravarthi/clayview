import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Mesh,
  type Material,
  type Matrix4,
  ShaderMaterial,
  Vector3,
} from "three";

const CHUNK_M = 72;

/**
 * VSM draws shadow receivers into the shadow map as well as casters, and there's no flag to stop
 * it. On a street that means a kilometre of road nobody needs, and curbs too thin for the map's
 * texels, which come out as a comb along the gutter. This puts every vertex outside clip space.
 */
const NO_SHADOW = new ShaderMaterial({
  vertexShader: "void main() { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }",
  fragmentShader: "void main() { gl_FragColor = vec4(1.0); }",
});
const A = new Vector3();
const B = new Vector3();
const C = new Vector3();
const AB = new Vector3();
const AC = new Vector3();

/** One growing mesh: position, normal and colour for every corner of every triangle. */
class Pile {
  readonly positions: number[] = [];
  readonly normals: number[] = [];
  readonly colors: number[] = [];

  corner(
    x: number,
    y: number,
    z: number,
    nx: number,
    ny: number,
    nz: number,
    color: Color,
  ): void {
    this.positions.push(x, y, z);
    this.normals.push(nx, ny, nz);
    this.colors.push(color.r, color.g, color.b);
  }

  build(): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(this.positions), 3));
    geometry.setAttribute("normal", new BufferAttribute(new Float32Array(this.normals), 3));
    geometry.setAttribute("color", new BufferAttribute(new Float32Array(this.colors), 3));
    return geometry;
  }
}

/**
 * Everything static in a world is poured in here and comes out as one merged mesh per patch of
 * ground, so a kilometre of street is a few dozen draw calls and frustum culling can skip most.
 * Coordinates are three.js axes in the shot's world frame: X = x, Y up, Z = -y.
 */
export class Soup {
  private readonly piles = new Map<string, Pile>();
  private pile: Pile | null = null;
  triangles = 0;

  /** Everything added until the next call lands in the patch holding this point. */
  at(x: number, z: number): this {
    const key = `${Math.floor(x / CHUNK_M)}:${Math.floor(z / CHUNK_M)}`;
    let pile = this.piles.get(key);
    if (!pile) {
      pile = new Pile();
      this.piles.set(key, pile);
    }
    this.pile = pile;
    return this;
  }

  private current(): Pile {
    if (!this.pile) throw new Error("call at() before adding to a soup");
    return this.pile;
  }

  /** A flat triangle, wound counter-clockwise from the outside. */
  triangle(ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number, color: Color): void {
    A.set(ax, ay, az);
    B.set(bx, by, bz);
    C.set(cx, cy, cz);
    AB.subVectors(B, A);
    AC.subVectors(C, A);
    AB.cross(AC).normalize();
    const pile = this.current();
    pile.corner(ax, ay, az, AB.x, AB.y, AB.z, color);
    pile.corner(bx, by, bz, AB.x, AB.y, AB.z, color);
    pile.corner(cx, cy, cz, AB.x, AB.y, AB.z, color);
    this.triangles++;
  }

  /** A flat quad a-b-c-d, wound counter-clockwise from the outside. */
  quad(a: Vector3, b: Vector3, c: Vector3, d: Vector3, color: Color): void {
    this.triangle(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, color);
    this.triangle(a.x, a.y, a.z, c.x, c.y, c.z, d.x, d.y, d.z, color);
  }

  /** One corner with its own normal and colour, for smooth strips. Add three at a time. */
  vertex(x: number, y: number, z: number, nx: number, ny: number, nz: number, color: Color): void {
    this.current().corner(x, y, z, nx, ny, nz, color);
  }

  countTriangles(count: number): void {
    this.triangles += count;
  }

  /** Pour in any geometry, baked to its place. Normals are taken from the geometry. */
  add(geometry: BufferGeometry, color: Color, matrix: Matrix4): void {
    const source = geometry.index ? geometry.toNonIndexed() : geometry;
    const position = source.getAttribute("position");
    const normal = source.getAttribute("normal");
    const shaded = source.getAttribute("color");
    const pile = this.current();
    const point = new Vector3();
    const direction = new Vector3();
    const local = new Color();
    for (let index = 0; index < position.count; index++) {
      point.fromBufferAttribute(position, index).applyMatrix4(matrix);
      direction.fromBufferAttribute(normal, index).transformDirection(matrix);
      if (shaded) local.setRGB(shaded.getX(index), shaded.getY(index), shaded.getZ(index));
      pile.corner(
        point.x,
        point.y,
        point.z,
        direction.x,
        direction.y,
        direction.z,
        shaded ? local.multiply(color) : color,
      );
    }
    this.triangles += position.count / 3;
  }

  meshes(material: Material, shadows: { cast: boolean; receive: boolean }): Mesh[] {
    const meshes: Mesh[] = [];
    for (const pile of this.piles.values()) {
      if (!pile.positions.length) continue;
      const mesh = new Mesh(pile.build(), material);
      mesh.castShadow = shadows.cast;
      mesh.receiveShadow = shadows.receive;
      if (!shadows.cast) mesh.customDepthMaterial = NO_SHADOW;
      mesh.geometry.computeBoundingSphere();
      mesh.matrixAutoUpdate = false;
      meshes.push(mesh);
    }
    this.piles.clear();
    this.pile = null;
    return meshes;
  }
}
