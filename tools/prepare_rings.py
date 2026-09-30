"""
Asset prep: turn the ring library in `3D Models/Rings STL/<Shape>/<Size>_<Shape>.stl`
(10 shapes x UK sizes H-Z, clean watertight meshes in mm) into what the viewer
loads.

  static/rings/<Shape>/<Size>.glb   one mesh per ring, viewer axes
  static/rings/catalog.json         every shape/size, for the ring menu
  static/rings/relief.json          the relief-capable ring's face region

Source axes: finger-hole axis = X, head up = +Z, hole centred on the origin.
Viewer axes: up = +Y, hole axis = Z, so (x, y, z)_viewer = (y, z, x)_source
(a cyclic swap, so handedness is kept).

The relief (heightmap) goes on one ring, RELIEF_SHAPE / RELIEF_SIZE. Its face
region is the ring's flat top: the triangles visible from straight above (a
top-down z-buffer) that tilt less than FACE_MAX_TILT_DEG from flat and connect
to the top. It stops where the top starts to round over, so the rounded edge
and the shoulders stay plain. That region is
refined REFINE_LEVELS times (each splits every triangle in 4) so the relief
has enough vertices for fine detail, with the neighbouring triangles split to
match so there are no cracks. Its vertices come first in the GLB, carry
top-down UVs (u along X, v along Z, image row 0 at -Z) spanning the region's
bounding box, and a `_WEIGHT` attribute that eases the relief out over the
last EDGE_BLEND_MM before the region's edge, where the surface turns vertical.

    python tools/prepare_rings.py            # everything
    python tools/prepare_rings.py --relief   # just the relief ring + json
"""
from __future__ import annotations

import argparse
import json
import struct
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
RELIEF_SHAPE, RELIEF_SIZE = "Square", "S"
REFINE_LEVELS = 2       # 0.17 mm source edges -> ~0.04 mm on the face
ZBUFFER_MM = 0.02       # top-down visibility raster resolution
EDGE_BLEND_MM = 0.2     # relief eases out over this distance before the region's edge
FACE_MAX_TILT_DEG = 4   # the relief stays on the flat top: it ends where the surface starts to tilt


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


def top_face(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """The ring's top face (viewer axes: up = +Y), as a bool mask of faces:
    visible from straight above and tilted less than FACE_MAX_TILT_DEG.

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
    flat = fn[:, 1] > np.cos(np.radians(FACE_MAX_TILT_DEG)) * np.linalg.norm(fn, axis=1)
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
    return keep


def refine(v: np.ndarray, f: np.ndarray, sel: np.ndarray, levels: int):
    """Split the selected faces 1-to-4, `levels` times, splitting neighbouring
    faces to match (1-to-2 or 1-to-3) so the mesh stays crack-free.
    Returns (vertices, faces, selected mask)."""
    for _ in range(levels):
        e_sel = np.sort(np.c_[f[sel][:, [0, 1]], f[sel][:, [1, 2]], f[sel][:, [2, 0]]].reshape(-1, 2), axis=1)
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
                out_sel += [False] * (2 if k == 1 else 3)
        f, sel = np.array(out_f, np.int64), np.array(out_sel, bool)
    return v, f, sel


def build_relief(v: np.ndarray, f: np.ndarray):
    """Refine the top-visible region and order it first. Returns (verts, faces,
    uv, weight, region_vertex_count, info)."""
    sel = top_face(v, f)
    print(f"  top face: {sel.sum()} of {len(f)} faces")
    v, f, sel = refine(v, f, sel, REFINE_LEVELS)
    print(f"  refined: {len(f)} faces ({sel.sum()} in the region), {len(v)} vertices")

    in_region = np.zeros(len(v), bool)
    in_region[np.unique(f[sel])] = True
    order = np.r_[np.where(in_region)[0], np.where(~in_region)[0]]
    remap = np.empty(len(v), np.int64)
    remap[order] = np.arange(len(v))
    v, f = v[order], remap[f]
    n_region = int(in_region.sum())

    # region boundary = edges used once by region faces
    rf = f[sel]
    e = np.sort(np.c_[rf[:, [0, 1]], rf[:, [1, 2]], rf[:, [2, 0]]].reshape(-1, 2), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    border = np.unique(uniq[cnt == 1])

    # bounding box and outline of the region seen from above
    rv = v[:n_region]
    x0, x1 = rv[:, 0].min(), rv[:, 0].max()
    z0, z1 = rv[:, 2].min(), rv[:, 2].max()
    fw, fd = x1 - x0, z1 - z0
    uv = np.c_[(v[:, 0] - x0) / fw, (v[:, 2] - z0) / fd]
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2

    # relief weight: eases out over EDGE_BLEND_MM before the region's edge
    bpts = v[border]
    dist = np.empty(n_region)
    for a in range(0, n_region, 4000):
        d2 = ((rv[a:a + 4000, None, :] - bpts[None, :, :]) ** 2).sum(-1)
        dist[a:a + 4000] = np.sqrt(d2.min(1))
    t = np.clip(dist / EDGE_BLEND_MM, 0, 1)
    weight = np.zeros(len(v))
    weight[:n_region] = t * t * (3 - 2 * t)

    # outline: the region's footprint from above, as a polygon (mm, centred on the bbox)
    res = 0.05
    W, H = int(fw / res) + 3, int(fd / res) + 3
    mask = np.zeros((H, W), np.uint8)
    px = np.round(np.c_[(v[:, 0] - x0) / res + 1, (v[:, 2] - z0) / res + 1]).astype(np.int32)
    for tri in f[sel]:
        cv2.fillConvexPoly(mask, px[tri], 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    outline = [[round((p[0] - 1) * res + x0 - cx, 3), round((p[1] - 1) * res + z0 - cz, 3)] for p in c]

    info = {
        "face_width_mm": round(float(fw), 3),
        "face_height_mm": round(float(fd), 3),
        "outline_xz_mm": outline,
        "region_vertex_count": n_region,
        "top_y_mm": round(float(rv[:, 1].max()), 3),
        "edge_blend_mm": EDGE_BLEND_MM,
        "uv": "u = (x - xmin) / face_width ; v = (z - zmin) / face_height ; image row 0 at -Z",
        "note": f"region = the flat top: visible from above and tilted < {FACE_MAX_TILT_DEG} deg",
    }
    return v, f, uv, weight, n_region, info


# --- main ------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relief", action="store_true", help="only rebuild the relief ring and relief.json")
    args = ap.parse_args()

    catalog = {"shapes": [], "relief": {"shape": RELIEF_SHAPE, "size": RELIEF_SIZE},
               "default": {"shape": RELIEF_SHAPE, "size": RELIEF_SIZE}}
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
            is_relief = folder == RELIEF_SHAPE and size == RELIEF_SIZE
            if args.relief and not is_relief:
                continue
            v, f = read_stl(src)
            v = to_viewer_axes(v)
            dia = inner_diameter(v, f)
            if is_relief:
                print(f"{folder} {size}: building relief face region")
                v, f, uv, w, n_region, info = build_relief(v, f)
                write_glb(OUT_DIR / rel, f"{size}_{folder}", v, f, {"TEXCOORD_0": uv, "_WEIGHT": w})
                info.update({"shape": folder, "size": size, "file": rel, "inner_diameter_mm": round(dia, 3)})
                (OUT_DIR / "relief.json").write_text(json.dumps(info))
            else:
                write_glb(OUT_DIR / rel, f"{size}_{folder}", v, f)
            ext = v.max(0) - v.min(0)
            entry["sizes"].append({"size": size, "file": rel, "inner_diameter_mm": round(dia, 2),
                                   "extent_mm": [round(float(x), 2) for x in ext]})
            print(f"{folder:20s} {size}: inner dia {dia:5.2f} mm -> {rel}")
        catalog["shapes"].append(entry)
    if not args.relief:
        (OUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=1))
        print(f"catalog: {sum(len(s['sizes']) for s in catalog['shapes'])} rings")


if __name__ == "__main__":
    main()
