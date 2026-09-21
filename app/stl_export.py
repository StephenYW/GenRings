"""
Stretch milestone: export a watertight STL of the face slab (flat base +
relief bumps from the 16-bit heightmap) using trimesh.

Geometry: a solid of constant thickness `base_thickness_mm`, with the top
surface height at each point equal to base_thickness_mm + relief_height(x, y).
The bottom is a flat rectangle at z=0. Side walls close the loop so the
result is a single watertight manifold.
"""
from __future__ import annotations

import numpy as np
import trimesh

from app import config


def heightmap_to_stl(
    heightmap_16bit: np.ndarray,
    face_width_mm: float = config.FACE_WIDTH_MM,
    face_height_mm: float = config.FACE_HEIGHT_MM,
    relief_max_mm: float = config.RELIEF_MAX_MM,
    base_thickness_mm: float = config.BASE_THICKNESS_MM,
    max_grid_dim: int = 220,
) -> trimesh.Trimesh:
    src_h, src_w = heightmap_16bit.shape
    stride = max(1, int(np.ceil(max(src_w, src_h) / max_grid_dim)))
    small = heightmap_16bit[::stride, ::stride]
    ny, nx = small.shape
    if nx < 2 or ny < 2:
        raise ValueError("Heightmap too small to build an STL grid")

    heights_mm = (small.astype(np.float64) / 65535.0) * relief_max_mm

    xs = np.linspace(-face_width_mm / 2, face_width_mm / 2, nx)
    ys = np.linspace(-face_height_mm / 2, face_height_mm / 2, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (ny, nx)

    top_z = base_thickness_mm + heights_mm
    top_verts = np.stack([grid_x, grid_y, top_z], axis=-1).reshape(-1, 3)
    bottom_verts = np.stack([grid_x, grid_y, np.zeros_like(top_z)], axis=-1).reshape(-1, 3)

    n_grid = nx * ny
    vertices = np.concatenate([top_verts, bottom_verts], axis=0)

    def top_idx(j, i):
        return j * nx + i

    def bottom_idx(j, i):
        return n_grid + j * nx + i

    faces = []

    # Top surface (normal +z): for each cell, two triangles, CCW seen from +z.
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = top_idx(j, i)
            v10 = top_idx(j, i + 1)
            v01 = top_idx(j + 1, i)
            v11 = top_idx(j + 1, i + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    # Bottom surface (normal -z): reverse winding relative to top.
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = bottom_idx(j, i)
            v10 = bottom_idx(j, i + 1)
            v01 = bottom_idx(j + 1, i)
            v11 = bottom_idx(j + 1, i + 1)
            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])

    # Side walls: walk the boundary loop (bottom row L->R, right col bottom->top,
    # top row R->L, left col top->bottom) and quad-stitch top to bottom, with
    # outward-facing winding.
    boundary = []
    boundary += [(0, i) for i in range(nx - 1)]              # bottom row, L->R
    boundary += [(j, nx - 1) for j in range(ny - 1)]          # right col, bottom->top
    boundary += [(ny - 1, i) for i in range(nx - 1, 0, -1)]   # top row, R->L
    boundary += [(j, 0) for j in range(ny - 1, 0, -1)]        # left col, top->bottom

    n_bound = len(boundary)
    for k in range(n_bound):
        j0, i0 = boundary[k]
        j1, i1 = boundary[(k + 1) % n_bound]
        t0, t1 = top_idx(j0, i0), top_idx(j1, i1)
        b0, b1 = bottom_idx(j0, i0), bottom_idx(j1, i1)
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.merge_vertices()
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh
