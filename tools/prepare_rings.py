"""
Asset prep: turn the ring library in `3D Models/Rings STL/<Shape>/<Size>_<Shape>.stl`
(10 shapes x UK sizes H-Z, clean watertight meshes in mm) into what the viewer
loads.

  static/rings/<Shape>/<Size>.glb   one mesh per ring, viewer axes
  static/rings/catalog.json         every shape/size, for the ring menu
  static/rings/relief.json          the design rings' design areas (per ring)

Source axes: finger-hole axis = X, head up = +Z, hole centred on the origin.
Viewer axes: up = +Y, hole axis = Z, so (x, y, z)_viewer = (y, z, x)_source
(a cyclic swap, so handedness is kept).

Designs (heightmaps) go on the rings in RELIEF_RINGS, each with its own way
of finding the design area:

- "recess" (S SquareRidged): the recessed floor inside the ridge -- the faces
  connected to the centre of the top that tilt less than RECESS_MAX_TILT_DEG,
  which takes in the floor and the small fillet where it meets the ridge's
  vertical inner wall, so the design fills all the space inside the ridge.
  That floor is refined REFINE_LEVELS times; its vertices and faces come
  first in the GLB. relief.json gives its box seen from above and outline.
- "tilt" (S Square), below.

The "tilt" design area is the ring's flat top: the triangles visible from straight above (a top-down z-buffer) that
tilt less than an angle the viewer sets live (a slider, default
DEFAULT_TILT_DEG, up to TILT_MAX_DEG). So the viewer can recompute that area
without a rebuild, the script prepares everything any angle could need:

- The "zone": the top faces up to TILT_MAX_DEG plus the band around them (the
  rounded edge and upper shoulders, within BLEND_MM, which the viewer bends to
  meet the design at the edge). It is refined REFINE_LEVELS times (each splits
  every triangle in 4), with the neighbouring triangles split to match so
  there are no cracks. In the GLB, the top faces come first, then the band's,
  then the rest; the zone's vertices likewise come first.
- relief.json's `tilt_table`: for every TILT_STEP_DEG, the flat top's bounding
  box seen from above (which the heightmap and crop box cover, and the viewer
  maps its top-down UVs to) and its outline.

    python tools/prepare_rings.py            # everything
    python tools/prepare_rings.py --relief   # just the relief ring + json
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "3D Models" / "Rings STL"
OUT_DIR = ROOT / "static" / "rings"

SIZES = list("HIJKLMNOPQRSTUVWXYZ")
SHAPES = [  # (folder, base shape label, ridged)
    ("Circle", "Circle", False), ("CircleRidged", "Circle", True),
    ("Oval", "Oval", False), ("OvalRidged", "Oval", True),
    ("Square", "Square", False), ("SquareRidged", "Square", True),
    ("Rectangle", "Rectangle", False), ("RectangleRidged", "Rectangle", True),
    ("ThinRectangle", "Thin rectangle", False), ("ThinRectangleRidged", "Thin rectangle", True),
]
# design rings: (shape folder, size) -> how its design area is found
RELIEF_RINGS = {("Square", "S"): "tilt", ("SquareRidged", "S"): "recess"}
DEFAULT_RING = ("Square", "S")
RECESS_MAX_TILT_DEG = 60  # recess: the floor + its fillet, up to where the ridge's inner wall turns vertical
REFINE_LEVELS = 2       # 0.17 mm source edges -> ~0.04 mm on the face
ZBUFFER_MM = 0.02       # top-down visibility raster resolution
DEFAULT_TILT_DEG = 8    # the design area: the top that tilts less than this (the viewer's slider starts here)
TILT_MAX_DEG = 30       # the slider's range; the mesh is prepared for any angle up to this
TILT_STEP_DEG = 0.5
BLEND_MM = 2.5          # how far down the band the curve that meets the design reaches
INNER_MM = 0.3          # on the flat top, the design eases into the (smoothed) edge height over this


# --- mesh io -------------------------------------------------------------------

def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (vertices, faces) with shared vertices merged."""
    with open(path, "rb") as fh:
        fh.read(80)
        n = int(np.frombuffer(fh.read(4), "<u4")[0])
        rec = np.frombuffer(fh.read(n * 50), count=n,
                            dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    verts, inv = np.unique(np.round(tri, 5), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3).astype(np.int64)


def to_viewer_axes(v: np.ndarray) -> np.ndarray:
    return v[:, [1, 2, 0]]


def write_glb(path: Path, name: str, positions: np.ndarray, faces: np.ndarray, extra: dict | None = None) -> None:
    """Minimal glTF 2.0 binary: one mesh, one triangle primitive. `extra` maps
    attribute names (e.g. TEXCOORD_0, _WEIGHT) to float arrays."""
    blobs, views, accessors, attributes = [], [], [], {}
    offset = 0

    def add(data: np.ndarray, target: int, acc: dict) -> int:
        nonlocal offset
        raw = data.tobytes()
        pad = (-len(raw)) % 4
        blobs.append(raw + b"\0" * pad)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        offset += len(raw) + pad
        accessors.append({"bufferView": len(views) - 1, **acc})
        return len(accessors) - 1

    pos = positions.astype(np.float32)
    attributes["POSITION"] = add(pos, 34962, {"componentType": 5126, "count": len(pos), "type": "VEC3",
                                              "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
    for key, arr in (extra or {}).items():
        arr = arr.astype(np.float32)
        kind = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[1 if arr.ndim == 1 else arr.shape[1]]
        attributes[key] = add(arr, 34962, {"componentType": 5126, "count": len(arr), "type": kind})
    idx = faces.astype(np.uint32).ravel()
    indices = add(idx, 34963, {"componentType": 5125, "count": len(idx), "type": "SCALAR"})

    gltf = {
        "asset": {"version": "2.0", "generator": "tools/prepare_rings.py"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attributes, "indices": indices, "mode": 4}]}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    body = b"".join(blobs)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(body), 0x004E4942) + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


# --- measurements --------------------------------------------------------------

def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


def inner_diameter(v: np.ndarray, f: np.ndarray) -> float:
    """Circle fit to the finger hole's inward-facing surface (viewer axes: hole along Z)."""
    n = vertex_normals(v, f)
    r = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
    inward = -(v[:, 0] * n[:, 0] + v[:, 1] * n[:, 1]) / r
    q = v[(inward > 0.95) & (np.abs(v[:, 2]) < 0.5)][:, :2]
    s = np.linalg.lstsq(np.c_[2 * q, np.ones(len(q))], (q ** 2).sum(1), rcond=None)[0]
    return float(2 * np.sqrt(s[2] + s[0] ** 2 + s[1] ** 2))


# --- relief region ---------------------------------------------------------------

def face_adjacency(f: np.ndarray) -> np.ndarray:
    """Pairs of faces sharing an edge."""
    e = np.sort(np.c_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]].reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(len(f)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = (e[1:] == e[:-1]).all(1)
    return np.c_[owner[:-1][same], owner[1:][same]]


def top_face(v: np.ndarray, f: np.ndarray, max_tilt_deg: float):
    """The ring's top face (viewer axes: up = +Y), as a bool mask of faces:
    visible from straight above and tilted less than max_tilt_deg. Also
    returns the z-buffer (face id per pixel, -1 = none) and its origin.

    Z-buffer in the XZ plane: draw every upward-facing triangle lowest first so
    each pixel keeps the highest one; any triangle left showing is visible.
    The result is then cleaned up to one connected patch without pinholes.
    """
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    up = fn[:, 1] > 0
    x0, z0 = v[:, 0].min() - 0.1, v[:, 2].min() - 0.1
    W = int((v[:, 0].max() + 0.1 - x0) / ZBUFFER_MM) + 1
    H = int((v[:, 2].max() + 0.1 - z0) / ZBUFFER_MM) + 1
    ids = -np.ones((H, W), np.int32)
    px = np.round(np.c_[(v[:, 0] - x0) / ZBUFFER_MM, (v[:, 2] - z0) / ZBUFFER_MM]).astype(np.int32)
    order = np.argsort(v[f].mean(1)[:, 1])
    for t in order[up[order]]:
        cv2.fillConvexPoly(ids, px[f[t]], int(t))
    vis = np.zeros(len(f), bool)
    vis[np.unique(ids[ids >= 0])] = True
    flat = fn[:, 1] > np.cos(np.radians(max_tilt_deg)) * np.linalg.norm(fn, axis=1)
    up = up & flat
    vis &= flat

    # keep the connected patch containing the highest face, and fill pinholes
    # (faces too small to win a pixel, surrounded by visible ones)
    adj = face_adjacency(f)
    for _ in range(3):
        cnt = np.zeros(len(f), int)
        np.add.at(cnt, adj[:, 0], vis[adj[:, 1]])
        np.add.at(cnt, adj[:, 1], vis[adj[:, 0]])
        vis |= up & (cnt >= 2)
    nbrs = [[] for _ in range(len(f))]
    for a, b in adj:
        if vis[a] and vis[b]:
            nbrs[a].append(b)
            nbrs[b].append(a)
    start = int(np.argmax(np.where(vis, v[f].mean(1)[:, 1], -np.inf)))
    keep = np.zeros(len(f), bool)
    keep[start] = True
    stack = [start]
    while stack:
        for nb in nbrs[stack.pop()]:
            if not keep[nb]:
                keep[nb] = True
                stack.append(nb)
    return keep, ids, (x0, z0)


def refine(v: np.ndarray, f: np.ndarray, sel: np.ndarray, levels: int):
    """Split the selected faces (sel != 0) 1-to-4, `levels` times, splitting
    neighbouring faces to match (1-to-2 or 1-to-3) so the mesh stays
    crack-free. `sel` may be a bool mask or integer labels; split faces keep
    their parent's label, the matching neighbour splits get 0.
    Returns (vertices, faces, labels)."""
    for _ in range(levels):
        fs = f[sel != 0]
        e_sel = np.sort(np.c_[fs[:, [0, 1]], fs[:, [1, 2]], fs[:, [2, 0]]].reshape(-1, 2), axis=1)
        e_sel = np.unique(e_sel, axis=0)
        mid_index = {(int(a), int(b)): len(v) + i for i, (a, b) in enumerate(e_sel)}
        v = np.vstack([v, (v[e_sel[:, 0]] + v[e_sel[:, 1]]) / 2])

        def mid(a, b):
            return mid_index.get((a, b) if a < b else (b, a), -1)

        out_f, out_sel = [], []
        for t, (a, b, c) in enumerate(f.tolist()):
            m = [mid(a, b), mid(b, c), mid(c, a)]
            k = sum(x >= 0 for x in m)
            if k == 0:
                out_f.append([a, b, c]); out_sel.append(sel[t])
            elif k == 3:
                ab, bc, ca = m
                out_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
                out_sel += [sel[t]] * 4
            else:
                # rotate so the split edges come first: (a,b) split, and (b,c) too when k == 2
                p, mm = [a, b, c], m
                while not (mm[0] >= 0 and (k == 1 or mm[1] >= 0)):
                    p, mm = p[1:] + p[:1], mm[1:] + mm[:1]
                a_, b_, c_ = p
                if k == 1:
                    out_f += [[a_, mm[0], c_], [mm[0], b_, c_]]
                else:
                    out_f += [[mm[0], b_, mm[1]], [a_, mm[0], mm[1]], [a_, mm[1], c_]]
                out_sel += [0] * (2 if k == 1 else 3)
        f, sel = np.array(out_f, np.int64), np.array(out_sel, dtype=sel.dtype)
    return v, f, sel


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Edges used by exactly one of `faces` (sorted vertex pairs)."""
    e = np.sort(np.c_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]].reshape(-1, 2), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]


def nearest(points: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the index of the nearest target and the distance to it."""
    idx = np.empty(len(points), np.int64)
    dist = np.empty(len(points))
    for a in range(0, len(points), 2000):
        d2 = ((points[a:a + 2000, None, :] - targets[None, :, :]) ** 2).sum(-1)
        idx[a:a + 2000] = d2.argmin(1)
        dist[a:a + 2000] = np.sqrt(d2[np.arange(len(d2)), idx[a:a + 2000]])
    return idx, dist


def blend_band(v: np.ndarray, f: np.ndarray, region: np.ndarray, depth: float) -> np.ndarray:
    """Faces of the band around `region` that the viewer bends to meet the
    design: connected to the region, within `depth` (+ a margin) of its edge,
    and not facing down (which keeps the finger hole's inner surface out)."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
    rim = np.unique(boundary_edges(f[region]))
    _, d = nearest(v[f].mean(1), v[rim])
    cand = ~region & (fn[:, 1] > -0.5) & (d < depth + 0.3)
    nbrs = [[] for _ in range(len(f))]
    for a, b in face_adjacency(f):
        nbrs[a].append(b)
        nbrs[b].append(a)
    band = np.zeros(len(f), bool)
    stack = list(np.where(region)[0])
    while stack:
        for nb in nbrs[stack.pop()]:
            if cand[nb] and not band[nb]:
                band[nb] = True
                stack.append(nb)
    return band


def tilt_table(f_tilt: np.ndarray, top: np.ndarray, ids: np.ndarray, origin) -> list[dict]:
    """For every TILT_STEP_DEG: the flat top's bounding box seen from above
    (viewer x/z, mm) and its outline (centred on the box), from the z-buffer."""
    x0r, z0r = origin
    table = []
    for deg in np.arange(TILT_STEP_DEG, TILT_MAX_DEG + 1e-9, TILT_STEP_DEG):
        sel = top & (f_tilt < deg)
        if not sel.any():
            continue
        mask = ((ids >= 0) & sel[np.maximum(ids, 0)]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        bx, bz, bw, bh = cv2.boundingRect(c)
        x0, z0 = x0r + bx * ZBUFFER_MM, z0r + bz * ZBUFFER_MM
        w, h = (bw - 1) * ZBUFFER_MM, (bh - 1) * ZBUFFER_MM
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)  # within 1 px = 0.02 mm
        cx, cz = x0 + w / 2, z0 + h / 2
        table.append({
            "deg": round(float(deg), 2),
            "x0": round(float(x0), 4), "z0": round(float(z0), 4),
            "width_mm": round(float(w), 3), "height_mm": round(float(h), 3),
            "outline_xz_mm": [[round(float(x0r + p[0] * ZBUFFER_MM - cx), 3),
                               round(float(z0r + p[1] * ZBUFFER_MM - cz), 3)] for p in c],
        })
    return table


def build_relief(v: np.ndarray, f: np.ndarray):
    """Prepare the relief ring for any design-area angle up to TILT_MAX_DEG:
    refine the zone (top faces + blend band), order it first, and tabulate
    the flat top's size and outline per angle. Returns (verts, faces, info)."""
    top, ids, origin = top_face(v, f, TILT_MAX_DEG)
    band = blend_band(v, f, top, BLEND_MM)
    print(f"  top (< {TILT_MAX_DEG} deg): {top.sum()} faces; blend band: {band.sum()} faces (of {len(f)})")
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    f_tilt = np.degrees(np.arccos(np.clip(fn[:, 1] / np.maximum(np.linalg.norm(fn, axis=1), 1e-12), -1, 1)))
    table = tilt_table(f_tilt, top, ids, origin)

    labels = np.where(top, 1, np.where(band, 2, 0)).astype(np.int8)
    v, f, labels = refine(v, f, labels, REFINE_LEVELS)
    # faces: top, then band, then the rest; vertices: the zone's first
    f = np.r_[f[labels == 1], f[labels == 2], f[labels == 0]]
    n_top_faces, n_zone_faces = int((labels == 1).sum()), int((labels != 0).sum())
    in_zone = np.zeros(len(v), bool)
    in_zone[np.unique(f[:n_zone_faces])] = True
    order = np.r_[np.where(in_zone)[0], np.where(~in_zone)[0]]
    remap = np.empty(len(v), np.int64)
    remap[order] = np.arange(len(v))
    v, f = v[order], remap[f]
    n_zone = int(in_zone.sum())
    print(f"  refined: {len(f)} faces ({n_top_faces} top, {n_zone_faces - n_top_faces} band), "
          f"{len(v)} vertices ({n_zone} in the zone); tilt table {len(table)} steps")

    default = min(table, key=lambda e: abs(e["deg"] - DEFAULT_TILT_DEG))
    info = {
        "mode": "tilt",
        "default_tilt_deg": default["deg"],
        "tilt_step_deg": TILT_STEP_DEG,
        # the default design area, for the backend's defaults
        "face_width_mm": default["width_mm"],
        "face_height_mm": default["height_mm"],
        "outline_xz_mm": default["outline_xz_mm"],
        "tilt_table": table,
        "top_face_count": n_top_faces,
        "zone_face_count": n_zone_faces,
        "zone_vertex_count": n_zone,
        "blend_mm": BLEND_MM,
        "inner_mm": INNER_MM,
        "uv": "top-down: u = (x - x0) / width_mm ; v = (z - z0) / height_mm ; image row 0 at -Z",
    }
    return v, f, info


def build_recess(v: np.ndarray, f: np.ndarray):
    """The recessed floor inside a ridged ring's ridge: refine it, order it
    first, and measure its box and outline from above. Returns (verts, faces, info)."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
    tilt = np.degrees(np.arccos(np.clip(fn[:, 1], -1, 1)))
    cen = v[f].mean(1)
    # start from the highest upward face over the ring's axis: the floor
    # (the finger hole's bottom also faces up there, but lower down)
    near = np.where((np.hypot(cen[:, 0], cen[:, 2]) < 0.5) & (fn[:, 1] > 0.9))[0]
    start = int(near[np.argmax(cen[near, 1])])
    ok = tilt < RECESS_MAX_TILT_DEG
    nbrs = [[] for _ in range(len(f))]
    for a, b in face_adjacency(f):
        nbrs[a].append(b)
        nbrs[b].append(a)
    floor = np.zeros(len(f), bool)
    floor[start] = True
    stack = [start]
    while stack:
        for nb in nbrs[stack.pop()]:
            if ok[nb] and not floor[nb]:
                floor[nb] = True
                stack.append(nb)
    print(f"  recessed floor: {floor.sum()} faces (of {len(f)})")

    v, f, floor = refine(v, f, floor, REFINE_LEVELS)
    f = np.r_[f[floor], f[~floor]]
    n_faces = int(floor.sum())
    in_region = np.zeros(len(v), bool)
    in_region[np.unique(f[:n_faces])] = True
    order = np.r_[np.where(in_region)[0], np.where(~in_region)[0]]
    remap = np.empty(len(v), np.int64)
    remap[order] = np.arange(len(v))
    v, f = v[order], remap[f]
    n_region = int(in_region.sum())

    rv = v[:n_region]
    x0, x1, z0, z1 = rv[:, 0].min(), rv[:, 0].max(), rv[:, 2].min(), rv[:, 2].max()
    w, h = x1 - x0, z1 - z0
    mask = np.zeros((int(h / ZBUFFER_MM) + 3, int(w / ZBUFFER_MM) + 3), np.uint8)
    px = np.round(np.c_[(v[:, 0] - x0) / ZBUFFER_MM + 1, (v[:, 2] - z0) / ZBUFFER_MM + 1]).astype(np.int32)
    for tri in f[:n_faces]:
        cv2.fillConvexPoly(mask, px[tri], 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    outline = [[round(float((p[0] - 1) * ZBUFFER_MM - w / 2), 3), round(float((p[1] - 1) * ZBUFFER_MM - h / 2), 3)] for p in c]
    print(f"  refined: {len(f)} faces ({n_faces} floor), {len(v)} vertices; floor {w:.2f} x {h:.2f} mm")
    info = {
        "mode": "recess",
        "face_width_mm": round(float(w), 3),
        "face_height_mm": round(float(h), 3),
        "x0": round(float(x0), 4), "z0": round(float(z0), 4),
        "outline_xz_mm": outline,
        "region_vertex_count": n_region,
        "region_face_count": n_faces,
        "max_tilt_deg": RECESS_MAX_TILT_DEG,
        "uv": "top-down: u = (x - x0) / face_width_mm ; v = (z - z0) / face_height_mm ; image row 0 at -Z",
    }
    return v, f, info


# --- main ------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relief", action="store_true", help="only rebuild the design rings and relief.json")
    args = ap.parse_args()

    # A build stamp: the viewer adds it to the GLB URLs so a browser never
    # pairs a cached model from an older build with this build's data.
    version = int(time.time())
    catalog = {"version": version, "shapes": [], "relief": [{"shape": sh, "size": sz} for sh, sz in RELIEF_RINGS],
               "default": {"shape": DEFAULT_RING[0], "size": DEFAULT_RING[1]}}
    reliefs = {}
    for folder, base, ridged in SHAPES:
        entry = {"id": folder, "base": base, "ridged": ridged, "sizes": []}
        for size in SIZES:
            src = SRC_DIR / folder / f"{size}_{folder}.stl"
            if not src.exists():
                # tolerate small naming slips (e.g. "V_Rectangle.Ridged.stl"):
                # take the one file in the folder for this size letter
                found = sorted((SRC_DIR / folder).glob(f"{size}_*.stl"))
                if len(found) != 1:
                    print(f"missing {src}")
                    continue
                src = found[0]
            rel = f"{folder}/{size}.glb"
            mode = RELIEF_RINGS.get((folder, size))
            is_relief = mode is not None
            if args.relief and not is_relief:
                continue
            v, f = read_stl(src)
            v = to_viewer_axes(v)
            dia = inner_diameter(v, f)
            if is_relief:
                print(f"{folder} {size}: building the design area ({mode})")
                v, f, info = build_relief(v, f) if mode == "tilt" else build_recess(v, f)
                write_glb(OUT_DIR / rel, f"{size}_{folder}", v, f)
                info.update({"shape": folder, "size": size, "file": rel, "inner_diameter_mm": round(dia, 3),
                             "vertex_count": len(v)})
                reliefs[f"{folder}/{size}"] = info
            else:
                write_glb(OUT_DIR / rel, f"{size}_{folder}", v, f)
            ext = v.max(0) - v.min(0)
            entry["sizes"].append({"size": size, "file": rel, "inner_diameter_mm": round(dia, 2),
                                   "extent_mm": [round(float(x), 2) for x in ext]})
            print(f"{folder:20s} {size}: inner dia {dia:5.2f} mm -> {rel}")
        catalog["shapes"].append(entry)
    (OUT_DIR / "relief.json").write_text(json.dumps({"version": version, "default": "/".join(DEFAULT_RING), "rings": reliefs}))
    if not args.relief:
        (OUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=1))
        print(f"catalog: {sum(len(s['sizes']) for s in catalog['shapes'])} rings")


if __name__ == "__main__":
    main()
