import { describe, expect, test } from "bun:test";
import { type BufferGeometry, MeshBasicMaterial } from "three";
import type { BuildingRun, RoadStation } from "../data/scene";
import { buildStreetLine, densify, StreetIndex } from "../data/street";
import { layMassing } from "./massing";
import { Soup } from "./soup";
import { layIslands, layPaint, laySurface } from "./surface";

function station(s: number, x: number, y: number, yaw: number): RoadStation {
  return {
    s,
    x,
    y,
    yaw,
    left: -5,
    right: 5,
    measured: true,
    leftCurb: true,
    rightCurb: true,
    leftWalk: 3,
    rightWalk: 3,
    leftGrass: 2,
    rightGrass: 2,
  };
}

function straight(count: number): RoadStation[] {
  return Array.from({ length: count }, (_, i) => station(i * 2, 0, i * 2, 0));
}

/** Straight, a U-turn of `radius`, then straight back alongside. */
function hairpin(radius: number): RoadStation[] {
  const road: RoadStation[] = [];
  let x = 0;
  let y = 0;
  let yaw = 0;
  const add = (): void => {
    road.push(station(road.length * 2, x, y, yaw));
    x += Math.sin(yaw) * 2;
    y += Math.cos(yaw) * 2;
  };
  for (let i = 0; i < 40; i++) add();
  const arc = Math.round((Math.PI * radius) / 2);
  for (let i = 0; i < arc; i++) {
    add();
    yaw += 2 / radius;
  }
  for (let i = 0; i < 40; i++) add();
  return road;
}

function geometries(soup: Soup): BufferGeometry[] {
  return soup
    .meshes(new MeshBasicMaterial(), { cast: false, receive: true })
    .map((mesh) => mesh.geometry);
}

/** Every triangle, as flat position triples. */
function triangles(geometry: BufferGeometry): number[][] {
  const position = geometry.getAttribute("position");
  const result: number[][] = [];
  for (let index = 0; index + 2 < position.count; index += 3) {
    const triangle: number[] = [];
    for (let corner = 0; corner < 3; corner++) {
      triangle.push(
        position.getX(index + corner),
        position.getY(index + corner),
        position.getZ(index + corner),
      );
    }
    result.push(triangle);
  }
  return result;
}

/** The Y of a triangle's winding normal: positive when it's front-facing from above. */
function facingUp([ax, , az, bx, , bz, cx, , cz]: number[]): number {
  const ux = bx! - ax!;
  const uz = bz! - az!;
  const vx = cx! - ax!;
  const vz = cz! - az!;
  return uz * vx - ux * vz;
}

describe("laySurface", () => {
  test("every flat triangle faces up, so none is culled away from above", () => {
    const soup = new Soup();
    laySurface(soup, densify(buildStreetLine(straight(60))!));
    let flat = 0;
    for (const geometry of geometries(soup)) {
      for (const triangle of triangles(geometry)) {
        const heights = [triangle[1]!, triangle[4]!, triangle[7]!];
        if (Math.max(...heights) - Math.min(...heights) > 1e-5) continue;
        const area = facingUp(triangle);
        if (Math.abs(area) < 1e-6) continue;
        flat++;
        expect(area).toBeGreaterThan(0);
      }
    }
    expect(flat).toBeGreaterThan(1000);
  });

  test("the sidewalk stands above the road, and the road above the ground", () => {
    const soup = new Soup();
    laySurface(soup, densify(buildStreetLine(straight(60))!));
    let highest = -Infinity;
    let lowest = Infinity;
    for (const geometry of geometries(soup)) {
      const position = geometry.getAttribute("position");
      for (let index = 0; index < position.count; index++) {
        highest = Math.max(highest, position.getY(index));
        lowest = Math.min(lowest, position.getY(index));
      }
    }
    expect(highest).toBeGreaterThan(0.1);
    expect(highest).toBeLessThan(0.25);
    expect(lowest).toBeLessThan(0);
  });
});

describe("layPaint", () => {
  test("a ring that can't be triangulated is skipped, the rest still land", () => {
    const soup = new Soup();
    const good = {
      kind: "stroke" as const,
      ring: [
        [0, 0],
        [1, 0],
        [1, 3],
        [0, 3],
      ] as [number, number][],
    };
    const degenerate = {
      kind: "figure" as const,
      ring: [
        [0, 0],
        [0, 0],
        [0, 0],
      ] as [number, number][],
    };
    expect(layPaint(soup, [degenerate, good])).toBe(1);
  });

  test("paint sits just above the road and faces up", () => {
    const soup = new Soup();
    layPaint(soup, [
      {
        kind: "stroke",
        ring: [
          [0, 0],
          [0, 3],
          [1, 3],
          [1, 0],
        ],
      },
    ]);
    const [geometry] = geometries(soup);
    for (const triangle of triangles(geometry!)) {
      expect(triangle[1]).toBeGreaterThan(0);
      expect(triangle[1]).toBeLessThan(0.05);
      expect(facingUp(triangle)).toBeGreaterThan(0);
    }
  });
});

describe("layIslands", () => {
  const index = new StreetIndex(buildStreetLine(straight(60))!);

  test("an upright on the carriageway gets an island, one beside it doesn't", () => {
    const soup = new Soup();
    const objects = [
      { kind: "sign" as const, x: 2, y: 40, height: 3 },
      { kind: "sign" as const, x: 7, y: 40, height: 3 },
    ];
    expect(layIslands(soup, objects, index)).toBe(1);
  });
});

describe("layMassing", () => {
  test("no block stands on the road where it doubles back", () => {
    const road = hairpin(12);
    const index = new StreetIndex(buildStreetLine(road)!);
    const run: BuildingRun = {
      side: "right",
      height: 12,
      points: Array.from({ length: 30 }, (_, i): [number, number] => [13, i * 3]),
    };
    const left: BuildingRun = {
      side: "left",
      height: 12,
      points: Array.from({ length: 30 }, (_, i): [number, number] => [-13, i * 3]),
    };
    const soup = new Soup();
    const blocks = layMassing(soup, [run, left], index);
    expect(blocks).toBeGreaterThan(2);
    for (const geometry of geometries(soup)) {
      const position = geometry.getAttribute("position");
      for (let at = 0; at < position.count; at += 7) {
        const place = index.place(position.getX(at), -position.getZ(at));
        expect(Math.abs(place.d)).toBeGreaterThan(5);
      }
    }
  });

  test("a building only glimpsed still comes out as a whole building", () => {
    const index = new StreetIndex(buildStreetLine(straight(60))!);
    const run: BuildingRun = {
      side: "right",
      height: 10,
      points: [
        [12, 40],
        [12, 44],
        [12, 47],
      ],
    };
    const soup = new Soup();
    expect(layMassing(soup, [run], index)).toBe(1);
    let low = Infinity;
    let high = -Infinity;
    for (const geometry of geometries(soup)) {
      const position = geometry.getAttribute("position");
      for (let at = 0; at < position.count; at++) {
        low = Math.min(low, -position.getZ(at));
        high = Math.max(high, -position.getZ(at));
      }
    }
    expect(high - low).toBeGreaterThanOrEqual(12.9);
  });
});
