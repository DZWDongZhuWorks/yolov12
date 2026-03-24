import numpy as np
import cv2

def _resample_polyline(line: np.ndarray, n_samples: int):
    if line.ndim != 2 or line.shape[0] < 2 or n_samples < 2:
        return line
    pts = line[:, :2].astype(np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-6:
        return line
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, int(n_samples))
    out = []
    j = 0
    for t in targets:
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        d = seg[j]
        if d <= 1e-6:
            out.append(pts[j].copy())
            continue
        ratio = float((t - cum[j]) / d)
        ratio = max(0.0, min(1.0, ratio))
        p = pts[j] * (1.0 - ratio) + pts[j + 1] * ratio
        out.append(p)
    return np.asarray(out, dtype=np.float32)

def extract_centerline_delaunay(arr: np.ndarray, step=15.0):
    if len(arr) < 3: return arr
    
    arr_closed = np.vstack([arr, arr[0]])
    poly_len = np.sum(np.linalg.norm(np.diff(arr_closed, axis=0), axis=1))
    n_samples = max(int(poly_len / step), 10)
    
    resampled = _resample_polyline(arr_closed, n_samples)[:-1]
    
    x_min, y_min = resampled.min(axis=0)
    x_max, y_max = resampled.max(axis=0)
    
    subdiv = cv2.Subdiv2D((int(x_min)-10, int(y_min)-10, int(x_max)+10, int(y_max)+10))
    for p in resampled:
        subdiv.insert((float(p[0]), float(p[1])))
        
    triangles = subdiv.getTriangleList()
    
    inside_tris = []
    arr_float32 = resampled.astype(np.float32)
    
    # 稍微退縮一個像素測試是否在內部，避免邊界誤差
    for t in triangles:
        p1 = np.array([t[0], t[1]])
        p2 = np.array([t[2], t[3]])
        p3 = np.array([t[4], t[5]])
        centroid = (p1 + p2 + p3) / 3.0
        
        # pointPolygonTest returns positive if inside
        if cv2.pointPolygonTest(arr_float32, (float(centroid[0]), float(centroid[1])), False) >= 0:
            inside_tris.append((p1, p2, p3, centroid))
            
    if not inside_tris:
        return arr
        
    def get_vertex_indices(tri_pts):
        diff = resampled[np.newaxis, :, :] - tri_pts[:, np.newaxis, :]
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
            
    # Find cycle first (for O-shape)
    visited = set()
    parent = {}
    cycle_path = []
    
    def dfs_cycle(start):
        stack = [(start, None)]
        while stack:
            curr, par = stack.pop()
            if curr in visited:
                # cycle detected
                path = [curr]
                c = par
                while c is not None and c != curr:
                    path.append(c)
                    c = parent.get(c)
                if c == curr:
                    path.append(curr)
                    return path
                continue
            visited.add(curr)
            parent[curr] = par
            for neighbor in adj[curr]:
                if neighbor != par:
                    stack.append((neighbor, curr))
        return []

    for i in range(n_nodes):
        if i not in visited:
            cycle = dfs_cycle(i)
            if len(cycle) > len(cycle_path):
                cycle_path = cycle
                
    main_path_indices = []
    if len(cycle_path) > max(n_nodes * 0.3, 5): # significant cycle
        print("Detected O-shape!")
        main_path_indices = cycle_path
    else:
        print("Detected U-shape or open shape!")
        # Find longest path via 2-BFS
        def bfs_farthest(start_node):
            q = [(start_node, [start_node])]
            vis = {start_node}
            farthest_node = start_node
            max_path = [start_node]
            while q:
                curr, path = q.pop(0)
                if len(path) > len(max_path):
                    max_path = path
                    farthest_node = curr
                for neighbor in adj[curr]:
                    if neighbor not in vis:
                        vis.add(neighbor)
                        q.append((neighbor, path + [neighbor]))
            return farthest_node, max_path
            
        start_node = 0
        farthest1, _ = bfs_farthest(start_node)
        _, longest_path = bfs_farthest(farthest1)
        main_path_indices = longest_path
        
    centerline = np.array([inside_tris[i][3] for i in main_path_indices])
    return centerline

if __name__ == "__main__":
    # Test U-shape
    u_shape = np.array([
        [10, 10], [10, 100], [50, 100], [50, 20], [70, 20], [70, 100], [110, 100], [110, 10]
    ], dtype=np.float32)
    center = extract_centerline_delaunay(u_shape)
    print("U-shape centerline points:", len(center))
    
    # Test O-shape
    o_shape = []
    # Outer circle
    for angle in np.linspace(0, 2*np.pi, 20, endpoint=False):
        o_shape.append([100 + 50*np.cos(angle), 100 + 50*np.sin(angle)])
    # Inner circle (backwards to create a hole if represented as one polygon, but usually yolov12 outputs one contour with a seam, or just the outer? Wait, if we use findContours, O shape might just be represented as a thick ribbon)
    # Let's create a ribbon O-shape by combining outer and inner with a zero-width seam
    seam = o_shape[0]
    inner_shape = []
    for angle in np.linspace(2*np.pi, 0, 20, endpoint=False):
        inner_shape.append([100 + 30*np.cos(angle), 100 + 30*np.sin(angle)])
    o_shape.extend([seam] + inner_shape)
    
    o_shape = np.array(o_shape, dtype=np.float32)
    center_o = extract_centerline_delaunay(o_shape)
    print("O-shape centerline points:", len(center_o))
