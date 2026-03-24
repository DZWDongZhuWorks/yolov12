import json
import numpy as np
import cv2
import sys
import os

from app_utils.polygon_utils import _resample_polyline

with open('test_demo/orignal_polygon.json', 'r') as f:
    data = json.load(f)
    
poly = data['objects'][0]['polygons'][0]
arr = np.array(poly, dtype=np.float32)

arr_closed = arr if np.allclose(arr[0], arr[-1]) else np.vstack([arr, arr[0]])
poly_len = float(np.sum(np.linalg.norm(np.diff(arr_closed, axis=0), axis=1)))

step = 15.0
n_samples = int(np.clip(poly_len / step, 10, 500))
resampled = _resample_polyline(arr_closed, n_samples)
resampled = resampled[:-1]

x_min, y_min = resampled.min(axis=0)
x_max, y_max = resampled.max(axis=0)
pad = 20.0
w = int(x_max - x_min + 2 * pad)
h = int(y_max - y_min + 2 * pad)
subdiv = cv2.Subdiv2D((int(x_min - pad), int(y_min - pad), w, h))

pts_inserted = []
for p in resampled:
    if not pts_inserted or float(np.linalg.norm(p - pts_inserted[-1])) > 0.1:
        subdiv.insert((float(p[0]), float(p[1])))
        pts_inserted.append(p)

triangles = subdiv.getTriangleList()
arr_float32 = arr_closed.astype(np.float32)

distances = []
for t in triangles:
    p1 = np.array([t[0], t[1]], dtype=np.float32)
    p2 = np.array([t[2], t[3]], dtype=np.float32)
    p3 = np.array([t[4], t[5]], dtype=np.float32)
    centroid = (p1 + p2 + p3) / 3.0
    dist = cv2.pointPolygonTest(arr_float32, (float(centroid[0]), float(centroid[1])), True)
    distances.append(dist)

dist_arr = np.array(distances)
neg_dists = dist_arr[dist_arr < 0]
print("All neg distances:", np.sort(neg_dists)[-20:]) # highest negative values
print("Just barely outside (-10 to 0):", len(neg_dists[neg_dists > -10.0]))

# Try counting components with tolerance
for tol in [0.0, -2.0, -5.0, -10.0]:
    inside_tris = []
    for i, t in enumerate(triangles):
        p1 = np.array([t[0], t[1]], dtype=np.float32)
        p2 = np.array([t[2], t[3]], dtype=np.float32)
        p3 = np.array([t[4], t[5]], dtype=np.float32)
        if distances[i] >= tol:
            inside_tris.append((p1, p2, p3))
            
    pts_inserted_np = np.asarray(pts_inserted, dtype=np.float32)
    def get_vertex_indices(tri_pts):
        diff = pts_inserted_np[np.newaxis, :, :] - tri_pts[:, np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        return tuple(sorted(np.argmin(dist_sq, axis=1).tolist()))
        
    edge_to_tri_idx = {}
    for i, t in enumerate(inside_tris):
        pts = np.array([t[0], t[1], t[2]])
        vid = get_vertex_indices(pts)
        if len(set(vid)) < 3:
            continue
        edges = [(vid[0], vid[1]), (vid[1], vid[2]), (vid[0], vid[2])]
        for e in edges:
            edge_to_tri_idx.setdefault(e, []).append(i)

    n_nodes = len(inside_tris)
    adj = {i: [] for i in range(n_nodes)}
    for e, t_indices in edge_to_tri_idx.items():
        if len(t_indices) == 2:
            u, v = t_indices
            adj[u].append(v)
            adj[v].append(u)

    visited_global = set()
    components = []
    for i in range(n_nodes):
        if i not in visited_global:
            comp = []
            q = [i]
            vis = {i}
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in vis:
                        vis.add(neighbor)
                        q.append(neighbor)
            visited_global.update(comp)
            components.append(comp)
            
    components.sort(key=len, reverse=True)
    print(f"Tolerance {tol:.1f}: {len(inside_tris)} tris, {len(components)} components. Sizes: {[len(c) for c in components[:5]]}")
