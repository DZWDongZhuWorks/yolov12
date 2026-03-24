import json
import numpy as np
import cv2
import sys
import os

# Import the function from polygon_utils
sys.path.append('d:/yolov12')
from app_utils.polygon_utils import _extract_centerline_delaunay, _resample_polyline

with open('test_demo/orignal_polygon.json', 'r') as f:
    data = json.load(f)
    
poly = data['objects'][0]['polygons'][0]
arr = np.array(poly, dtype=np.float32)

print(f"Original polygon points: {len(arr)}")

# Run the algorithm step by step to find the issue
arr_closed = arr if np.allclose(arr[0], arr[-1]) else np.vstack([arr, arr[0]])
diffs = np.diff(arr_closed, axis=0)
seg_lens = np.linalg.norm(diffs, axis=1)
poly_len = float(np.sum(seg_lens))
print(f"Perimeter: {poly_len}")

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

inside_tris = []
for t in triangles:
    p1 = np.array([t[0], t[1]], dtype=np.float32)
    p2 = np.array([t[2], t[3]], dtype=np.float32)
    p3 = np.array([t[4], t[5]], dtype=np.float32)
    centroid = (p1 + p2 + p3) / 3.0
    if cv2.pointPolygonTest(arr_float32, (float(centroid[0]), float(centroid[1])), False) >= 0:
        inside_tris.append((p1, p2, p3, centroid))

print(f"Inside triangles: {len(inside_tris)}")

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

# Find connected components to see if the graph breaks
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
print(f"Number of components: {len(components)}")
for idx, c in enumerate(components[:5]):
    print(f"Component {idx}: size {len(c)}")

# Check Y ranges of the largest component
largest_comp = components[0]
y_vals = [inside_tris[i][3][1] for i in largest_comp]
print(f"Largest component Y range: {min(y_vals):.1f} to {max(y_vals):.1f}")

if len(components) > 1:
    comp2 = components[1]
    y_vals2 = [inside_tris[i][3][1] for i in comp2]
    print(f"Second component Y range: {min(y_vals2):.1f} to {max(y_vals2):.1f}")
