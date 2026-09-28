"""
One-off asset prep: turn the raw Meshy ring GLB into a viewer-ready GLB whose
entire head top is a clean, smooth, UV-mapped `face` mesh for the relief.

The source head top is one smooth pillow-shaped dome that rolls over into
steep sides (no flat plateau), with faceted AI-mesh noise. So:

  1. The body is cut at a plane just below the dome's edge (CUT_Y). Everything
     above the cut -- the whole top of the head -- is discarded.
  2. A smooth polynomial height field is fitted to the discarded dome. It has
     the dome's shape without the facet noise, and analytic normals.
  3. A fresh, evenly tessellated `face` mesh is built over the cut outline from
     that height field, pinned to the body's cut edge so there is no seam.

The relief covers the whole face, out to the rim where it meets the shoulders.
The viewer displaces `face` vertices straight up (+Y) by the heightmap and bends
the body's shoulder up to meet the displaced edge, so the shoulder rises to meet
the textured top and its edge follows the texture. The smoothed rim normals
(`rim_normals`) are still exported to ring_face.json and set on the body's rim
vertices, though the viewer now recomputes normals after bending.

Output (viewer coordinates, millimetres): hole axis = Z, up = +Y, face-outline
bounding-box centre at x = z = 0, cut plane at y = 0 (dome crest is above it).
The first N vertices of the `face` mesh are the rim loop, in order, at the same
positions as the body's cut edge.

    python tools/prepare_ring.py

Outputs static/models/ring.glb and static/models/ring_face.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "3D Models" / "Meshy_AI_Silver_Signet_Ring_0925210847_generate.glb"
OUT_DIR = ROOT / "static" / "models"

INNER_DIAMETER_MM = 18.0
# Cut height in the source's normalised units (dome crest is ~0.950). At
# ~0.82 the dome has rolled over into the steep sides, so everything above
# is 'top'. Lower values start eating into the vertical sides/shoulders.
CUT_Y = 0.82
FACE_GRID_MM = 0.07  # interior point spacing of the generated face mesh
POLY_DEGREE = 8      # smooth height-field fit to the dome


def measure_inner_diameter(mesh: trimesh.Trimesh) -> float:
    """Fit a circle to the band's inward-facing surface (loop lies in XY)."""
    v = mesh.vertices
    n = mesh.vertex_normals
    r = np.maximum(np.linalg.norm(v[:, :2], axis=1), 1e-9)
    inward = -(v[:, 0] * n[:, 0] + v[:, 1] * n[:, 1]) / r
    sel = (inward > 0.9) & (np.abs(v[:, 2]) < 0.3)
    q = v[sel][:, :2]
    a = np.c_[2 * q, np.ones(len(q))]
    sol = np.linalg.lstsq(a, (q**2).sum(1), rcond=None)[0]
    radius = np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)
    return 2 * float(radius)


def clip_below_plane(vertices: np.ndarray, faces: np.ndarray, y0: float):
    """Keep the part of the mesh with y <= y0, cutting triangles exactly on
    the plane. Returns (vertices, faces, cut_vertex_indices)."""
    verts = [tuple(p) for p in vertices]
    cache: dict[tuple[int, int], int] = {}
    cut_ids: set[int] = set()

    def cross(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in cache:
            return cache[key]
        a, b = vertices[i], vertices[j]
        t = (y0 - a[1]) / (b[1] - a[1])
        p = a + t * (b - a)
        p[1] = y0
        verts.append(tuple(p))
        cache[key] = len(verts) - 1
        cut_ids.add(len(verts) - 1)
        return cache[key]

    out = []
    above = vertices[:, 1] > y0
    for f in faces:
        flags = above[f]
        k = int(flags.sum())
        if k == 0:
            out.append(tuple(f))
        elif k == 3:
            continue
        elif k == 1:
            # one vertex above: quad below -> two triangles
            i = int(np.where(flags)[0][0])
            p, q, r = f[i], f[(i + 1) % 3], f[(i + 2) % 3]
            pq, pr = cross(p, q), cross(p, r)
            out += [(pq, q, r), (pq, r, pr)]
        else:
            # two above: one triangle below
            i = int(np.where(~flags)[0][0])
            p, q, r = f[i], f[(i + 1) % 3], f[(i + 2) % 3]
            out.append((p, cross(p, q), cross(p, r)))
    return np.array(verts), np.array(out, dtype=np.int64), cut_ids


def boundary_loop(faces: np.ndarray, allowed: set[int]) -> list[int]:
    """Ordered loop of the open boundary edges whose endpoints are cut vertices."""
    edges: dict[tuple[int, int], int] = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(a, b), max(a, b))
            edges[k] = edges.get(k, 0) + 1
    adj: dict[int, list[int]] = {}
    for (a, b), c in edges.items():
        if c == 1 and a in allowed and b in allowed:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    if not adj or any(len(v) != 2 for v in adj.values()):
        raise RuntimeError("cut boundary is not a single clean loop")
    start = next(iter(adj))
    loop, prev, cur = [start], None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        nxt = nxt[0] if prev is not None else adj[cur][0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    if len(loop) != len(adj):
        raise RuntimeError("cut boundary has more than one loop")
    return loop


def rasterize_dome(mesh: trimesh.Trimesh, extent=(-0.85, 0.85, -0.7, 0.7), width=1600):
    """Top-down height raster (source units) of the upward-facing head top."""
    v, f = mesh.vertices, mesh.faces
    fn = mesh.face_normals
    cen = v[f].mean(1)
    sel = np.where((cen[:, 1] > 0.6) & (fn[:, 1] > 0.0))[0]
    sel = sel[np.argsort(cen[sel, 1])]  # low -> high so the top overwrites
    x0, x1, z0, z1 = extent
    sc = width / (x1 - x0)
    hgt = np.zeros((int((z1 - z0) * sc), width), np.float32)
    for i in sel:
        poly = np.array([[(p[0] - x0) * sc, (p[2] - z0) * sc] for p in v[f[i]]], np.int32)
        cv2.fillConvexPoly(hgt, poly, float(cen[i, 1]))
    return hgt, sc, (x0, z0)


def fit_dome(hgt: np.ndarray, sc: float, origin, cut_y: float, degree: int):
    """Least-squares 2D polynomial fit to the dome (pixels at/above cut_y).
    Returns (height(x, z), gradient(x, z)) in source units."""
    region = (hgt >= cut_y).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(region)
    region = lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    rows, cols = np.nonzero(region)
    pick = np.random.default_rng(0).choice(len(rows), min(len(rows), 120_000), replace=False)
    x = cols[pick] / sc + origin[0]
    z = rows[pick] / sc + origin[1]
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = lambda x, z: np.stack([(x / 0.85) ** i * (z / 0.7) ** j for i, j in terms], 1)
    coef = np.linalg.lstsq(A(x, z), hgt[rows[pick], cols[pick]].astype(float), rcond=None)[0]

    def height(x, z):
        return A(x, z) @ coef

    def gradient(x, z):
        gx = np.stack([i * (x / 0.85) ** max(i - 1, 0) * (z / 0.7) ** j / 0.85 if i else 0 * x
                       for i, j in terms], 1)
        gz = np.stack([j * (x / 0.85) ** i * (z / 0.7) ** max(j - 1, 0) / 0.7 if j else 0 * x
                       for i, j in terms], 1)
        return gx @ coef, gz @ coef

    return height, gradient


def build_face(loop_xz: np.ndarray, grid_mm: float):
    """Delaunay-triangulate the polygon (boundary vertices + interior grid)."""
    poly = loop_xz.astype(np.float32)
    x0, z0 = poly.min(0)
    x1, z1 = poly.max(0)
    xs = np.arange(x0, x1, grid_mm)
    zs = np.arange(z0, z1, grid_mm)
    gx, gz = np.meshgrid(xs, zs)
    grid = np.c_[gx.ravel(), gz.ravel()].astype(np.float32)
    contour = poly.reshape(-1, 1, 2)
    keep = np.array([cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True) > grid_mm * 0.6 for p in grid])
    pts = np.vstack([poly, grid[keep]])

    pad = 1.0
    sub = cv2.Subdiv2D((float(x0 - pad), float(z0 - pad), float(x1 - x0 + 2 * pad), float(z1 - z0 + 2 * pad)))
    for p in pts:
        sub.insert((float(p[0]), float(p[1])))
    index = {(round(float(p[0]), 4), round(float(p[1]), 4)): i for i, p in enumerate(pts)}
    tris = []
    for t in sub.getTriangleList():
        corners = [(round(float(t[i]), 4), round(float(t[i + 1]), 4)) for i in (0, 2, 4)]
        if not all(c in index for c in corners):
            continue  # touches Subdiv2D's virtual outer vertices
        c = np.mean(corners, axis=0)
        if cv2.pointPolygonTest(contour, (float(c[0]), float(c[1])), False) < 0:
            continue
        tris.append([index[c_] for c_ in corners])
    return pts, np.array(tris, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-y", type=float, default=CUT_Y)
    ap.add_argument("--inner-diameter", type=float, default=INNER_DIAMETER_MM)
    args = ap.parse_args()

    src = trimesh.load(SRC, force="mesh", process=False)
    scale = args.inner_diameter / measure_inner_diameter(src)
    crest = float(src.vertices[:, 1].max())
    print(f"scale {scale:.4f} mm per source unit")

    # 1. Smooth fit to the dome, in source units.
    hgt, sc, origin = rasterize_dome(src)
    height, gradient = fit_dome(hgt, sc, origin, args.cut_y, POLY_DEGREE)

    # 2. Body: everything below the cut plane.
    verts, faces, cut_ids = clip_below_plane(np.asarray(src.vertices, float), np.asarray(src.faces), args.cut_y)
    loop = boundary_loop(faces, cut_ids)
    print(f"cut loop: {len(loop)} vertices")

    lp_src = verts[loop]
    cx = (lp_src[:, 0].min() + lp_src[:, 0].max()) / 2
    cz = (lp_src[:, 2].min() + lp_src[:, 2].max()) / 2
    to_mm = lambda p: (p - np.array([cx, args.cut_y, cz])) * scale
    verts_mm = to_mm(verts)
    loop_xz = verts_mm[loop][:, [0, 2]]

    # 3. Face: Delaunay over (cut-loop vertices + interior grid), heights from
    #    the fitted dome plus a smooth correction so the loop stays exactly on
    #    the body's cut edge (residual interpolated by inverse-distance).
    pts, face_tris = build_face(loop_xz, FACE_GRID_MM)
    fx, fz = pts[:, 0] / scale + cx, pts[:, 1] / scale + cz
    fy = height(fx, fz)
    loop_res = args.cut_y - height(lp_src[:, 0], lp_src[:, 2])
    nb = len(loop)
    corr = np.empty(len(pts))
    for a in range(0, len(pts), 4000):
        d2 = ((pts[a:a + 4000, None, :] - loop_xz[None, :, :]) ** 2).sum(-1) + 1e-4
        w = 1.0 / d2
        corr[a:a + 4000] = (w * loop_res[None, :]).sum(1) / w.sum(1)
    fy = fy + corr
    fy[:nb] = args.cut_y  # boundary vertices are exactly the body's
    face_verts = np.c_[pts[:, 0], (fy - args.cut_y) * scale, pts[:, 1]]
    gx, gz = gradient(fx, fz)
    nrm = np.c_[-gx, np.ones_like(gx), -gz]
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]

    # The relief covers the whole face; UVs span its bounding box.
    x0, z0 = loop_xz.min(0)
    x1, z1 = loop_xz.max(0)
    fw, fd = x1 - x0, z1 - z0
    u = (face_verts[:, 0] - x0) / fw
    v = (face_verts[:, 2] - z0) / fd  # image row 0 (top) = -Z

    a_, b_, c_ = (face_verts[face_tris[:, i]] for i in range(3))
    flip = np.cross(b_ - a_, c_ - a_)[:, 1] < 0  # wind faces so they point up
    face_tris[flip] = face_tris[flip][:, ::-1]

    # Rim normals: the body's own wall normals along the cut edge, smoothed
    # along the loop, and set on the body's rim vertices.
    body_full = trimesh.Trimesh(verts_mm, faces, process=False)
    bn = np.array(body_full.vertex_normals)
    rim_n = bn[loop]
    k = 6
    rim_n = sum(np.roll(rim_n, i, axis=0) for i in range(-k, k + 1))
    rim_n /= np.linalg.norm(rim_n, axis=1)[:, None]
    bn[loop] = rim_n
    used = np.unique(faces)
    remap = -np.ones(len(verts_mm), dtype=np.int64)
    remap[used] = np.arange(len(used))
    body = trimesh.Trimesh(verts_mm[used], remap[faces], vertex_normals=bn[used], process=False)
    face = trimesh.Trimesh(
        face_verts, face_tris, process=False, vertex_normals=nrm,
        visual=trimesh.visual.TextureVisuals(uv=np.c_[u, v]),
    )

    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body", geom_name="body")
    scene.add_geometry(face, node_name="face", geom_name="face")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ring.glb").write_bytes(scene.export(file_type="glb"))

    info = {
        "units": "mm",
        "inner_diameter_mm": args.inner_diameter,
        # face_width/height_mm and outline_xz_mm describe the area the heightmap
        # covers (the whole face), centred on its own bounding box.
        "face_width_mm": round(float(fw), 3),
        "face_height_mm": round(float(fd), 3),
        "outline_xz_mm": [[round(float(x), 3), round(float(z), 3)] for x, z in loop_xz],
        "rim_normals": [[round(float(a), 4) for a in n] for n in rim_n],
        "surface_area_mm2": round(float(face.area), 2),
        "dome_rise_above_cut_mm": round(float((crest - args.cut_y) * scale), 3),
        "axes": {"hole_axis": "z", "up": "+y", "cut_plane_y": 0},
        "uv": "u = (x - xmin) / face_width ; v = (z - zmin) / face_height ; image row 0 at -Z",
        "rim": "face vertices 0..len(outline)-1 are the rim loop, in order (outline_xz_mm order)",
    }
    (OUT_DIR / "ring_face.json").write_text(json.dumps(info))
    print(f"face {fw:.2f} x {fd:.2f} mm, {len(face_tris)} tris; "
          f"body {len(body.faces)} tris; fit-vs-cut correction max {np.abs(loop_res).max() * scale:.3f} mm")


if __name__ == "__main__":
    main()
