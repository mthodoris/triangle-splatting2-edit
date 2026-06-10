import sys
from collections import defaultdict


def load_obj(path):
    vertices = []
    faces = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == 'f':
                # support v, v/vt, v/vt/vn, v//vn formats
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(tuple(indices))
    return vertices, faces


def analyze_mesh(path):
    vertices, faces = load_obj(path)

    print(f"=== Mesh: {path} ===")
    print(f"Vertices : {len(vertices)}")
    print(f"Faces    : {len(faces)}")

    # vertex -> list of face indices that use it
    vertex_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for vi in face:
            vertex_to_faces[vi].append(fi)

    valences = [len(vertex_to_faces[vi]) for vi in range(len(vertices))]
    isolated = sum(1 for v in valences if v == 0)
    shared    = sum(1 for v in valences if v > 1)

    print(f"\n--- Vertex sharing ---")
    print(f"Isolated vertices (0 faces)  : {isolated}")
    print(f"Vertices in exactly 1 face   : {sum(1 for v in valences if v == 1)}")
    print(f"Vertices in >1 face (shared) : {shared}")
    print(f"Max faces per vertex         : {max(valences) if valences else 0}")
    print(f"Avg faces per vertex         : {sum(valences)/len(valences):.2f}" if valences else "N/A")

    # distribution
    from collections import Counter
    dist = Counter(valences)
    print("\n  valence : count")
    for k in sorted(dist):
        print(f"  {k:7d} : {dist[k]}")

    # edge connectivity: edge -> list of face indices
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            e = tuple(sorted((face[i], face[(i + 1) % n])))
            edge_to_faces[e].append(fi)

    boundary_edges   = sum(1 for f in edge_to_faces.values() if len(f) == 1)
    manifold_edges   = sum(1 for f in edge_to_faces.values() if len(f) == 2)
    nonmanifold_edges = sum(1 for f in edge_to_faces.values() if len(f) > 2)

    print(f"\n--- Edge connectivity ---")
    print(f"Total unique edges   : {len(edge_to_faces)}")
    print(f"Boundary edges (1 face)    : {boundary_edges}")
    print(f"Manifold edges (2 faces)   : {manifold_edges}")
    print(f"Non-manifold edges (>2)    : {nonmanifold_edges}")

    # connected components via union-find on vertices
    parent = list(range(len(vertices)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for face in faces:
        for i in range(1, len(face)):
            union(face[0], face[i])

    roots = set(find(v) for v in range(len(vertices)) if valences[v] > 0)
    print(f"\n--- Connectivity ---")
    print(f"Connected components : {len(roots)}")

    print(f"\n--- Face valence ---")
    face_sizes = Counter(len(f) for f in faces)
    for k in sorted(face_sizes):
        label = {3: "triangles", 4: "quads"}.get(k, f"{k}-gons")
        print(f"  {label}: {face_sizes[k]}")

    # triangle adjacency: number of shared edges per triangle
    tri_neighbors = defaultdict(set)
    for e, flist in edge_to_faces.items():
        if len(flist) == 2:
            a, b = flist
            tri_neighbors[a].add(b)
            tri_neighbors[b].add(a)

    neighbor_counts = [len(tri_neighbors[fi]) for fi in range(len(faces))]
    isolated_tris = sum(1 for c in neighbor_counts if c == 0)
    fully_connected = sum(1 for c in neighbor_counts if c == 3)

    print(f"\n--- Triangle adjacency (shared edges) ---")
    print(f"Isolated triangles (0 neighbors) : {isolated_tris}")
    print(f"Fully connected    (3 neighbors) : {fully_connected}")
    print(f"Max neighbors per triangle       : {max(neighbor_counts) if neighbor_counts else 0}")
    print(f"Avg neighbors per triangle       : {sum(neighbor_counts)/len(neighbor_counts):.2f}" if neighbor_counts else "N/A")

    adj_dist = Counter(neighbor_counts)
    print("\n  neighbors : count")
    for k in sorted(adj_dist):
        print(f"  {k:9d} : {adj_dist[k]}")

    # triangles connected through shared vertices (includes edge-sharing)
    tri_vert_neighbors = defaultdict(set)
    for vi, flist in vertex_to_faces.items():
        for a in flist:
            for b in flist:
                if a != b:
                    tri_vert_neighbors[a].add(b)

    connected_via_vertex = sum(1 for fi in range(len(faces)) if len(tri_vert_neighbors[fi]) > 0)
    pct = 100.0 * connected_via_vertex / len(faces) if faces else 0.0
    print(f"\n--- Vertex-connected triangles ---")
    print(f"Triangles sharing >=1 vertex with another : {connected_via_vertex} / {len(faces)} ({pct:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_mesh.py <mesh.obj>")
        sys.exit(1)
    analyze_mesh(sys.argv[1])
