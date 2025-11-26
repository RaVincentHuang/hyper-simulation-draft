from dependency import LocalDoc, Node, Pos, Relationship


class Vertex:
    def __init__(self, id: int, nodes: list[Node]):
        self.id = id
        self.nodes = nodes
        
    def pos_equal(self, pos: Pos) -> bool:
        if not self.nodes:
            return False
        return any(n.pos == pos for n in self.nodes)
        
    def pos_same(self, other: 'Vertex') -> bool:
        if not self.nodes or not other.nodes:
            return False
        return any(n1.pos == n2.pos for n1, n2 in zip(self.nodes, other.nodes))
    
    def text(self) -> str:
        return self.nodes[0].text if self.nodes else ""
    
    @staticmethod
    def from_nodes(vertices: list[Node], id_map: dict[Node, int]) -> list['Vertex']:
        vertex_map: dict[int, list[Node]] = {}
        for vertex in vertices:
            vid = id_map.get(vertex)
            if vid is None:
                continue
            if vid not in vertex_map:
                vertex_map[vid] = []
            vertex_map[vid].append(vertex)
        return [Vertex(vid, nodes) for vid, nodes in vertex_map.items()]
    
    @staticmethod
    def vertex_node_map(vertices: list['Vertex']) -> dict[Node, 'Vertex']:
        vertex_map: dict[Node, Vertex] = {}
        for vertex in vertices:
            for node in vertex.nodes:
                vertex_map[node] = vertex
        return vertex_map

class Hyperedge:
    def __init__(self, root: Vertex, vertices: list[Vertex], desc: str, full_desc: str, start: int, end: int):
        self.root = root
        self.vertices = vertices
        self.desc = desc
        self.full_desc = full_desc
        self.start = start
        self.end = end
        
        self.father: Hyperedge | None = None
        
    def current_node(self, vertex: Vertex) -> Node:
        for node in vertex.nodes:
            # print(f"node index is {node.index}, hyperedge range is {self.start}-{self.end}")
            if node.index >= self.start and node.index <= self.end:
                return node
        assert False, f"Vertex does not contain a node in hyperedge range, Vertex nodes: {vertex.nodes}, Hyperedge range: {self.start}-{self.end}, Hyperedge is {self.desc}"

    @staticmethod
    def form_relationship(relationship: Relationship, vertex_map: dict[Node, Vertex]) -> 'Hyperedge':
        vertices = []
        root = vertex_map.get(relationship.root)
        assert root is not None, f"Root vertex not found in vertex map. Relationship root: {relationship.root}"
        for node in relationship.entities:
            vertex = vertex_map.get(node)
            assert vertex is not None, f"Entity vertex not found in vertex map. Entity node: {node}"
            # if vertex and vertex not in vertices:
            if vertex not in vertices:
                vertices.append(vertex)
        return Hyperedge(root, vertices, relationship.relationship_text_simple(), relationship.sentence, relationship.start, relationship.end)

    def __format__(self, format_spec: str) -> str:
        return f"Hyperedge(desc={self.desc}, vertices={[v.id for v in self.vertices]})"

class Path:
    def __init__(self, hyperedges: list[Hyperedge]) -> None:
        self.hyperedges: list[Hyperedge] = hyperedges
    
    def length(self) -> int:
        return len(self.hyperedges)
    

class Hypergraph:
    def __init__(self, vertices: list[Vertex], hyperedges: list[Hyperedge], doc: LocalDoc) -> None:
        self.vertices = vertices
        self.hyperedges = hyperedges
        self.doc = doc
        self.contained_edges: dict[Vertex, list[Hyperedge]] = {}
        for hyperedge in self.hyperedges:
            for vertex in hyperedge.vertices:
                if vertex not in self.contained_edges:
                    self.contained_edges[vertex] = []
                self.contained_edges[vertex].append(hyperedge)
        self.path_map_cache: dict[tuple[Vertex, Vertex], list[Path]] = {}
        self.neighbor_map_cache: dict[int, dict[Vertex, set[Vertex]]] = {}
    
    @staticmethod
    def from_rels(vertices: list[Node], relationships: list[Relationship], id_map: dict[Node, int], doc: LocalDoc) -> 'Hypergraph':
        vertex_objs = Vertex.from_nodes(vertices, id_map)
        vertex_map = Vertex.vertex_node_map(vertex_objs)
        hyperedges = []
        rel_to_hyperedge: dict[Relationship, Hyperedge] = {}
        for rel in relationships:
            hyperedge = Hyperedge.form_relationship(rel, vertex_map)
            rel_to_hyperedge[rel] = hyperedge
            hyperedges.append(hyperedge)
        
        # set father hyperedge for each hyperedge
        for rel, hyperedge in rel_to_hyperedge.items():
            if rel.father:
                father_hyperedge = rel_to_hyperedge.get(rel.father)
                if father_hyperedge:
                    hyperedge.father = father_hyperedge
            print(f"Hyperedge: {hyperedge.desc}, Father: {hyperedge.father.desc if hyperedge.father else None}")
        return Hypergraph(vertex_objs, hyperedges, doc)
    
    # use pickle to serialize and deserialize Hypergraph
    def save(self, filepath: str) -> None:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'Hypergraph':
        import pickle
        with open(filepath, 'rb') as f:
            hypergraph = pickle.load(f)
        return hypergraph
    
    def neighbors(self, vertex: Vertex, hop: int=-1) -> set[Vertex]:
        # if hop == -1, return all reachable neighbors
        if hop not in self.neighbor_map_cache:
            self.neighbor_map_cache[hop] = self._build_neighbors_map(hop)
        return self.neighbor_map_cache[hop].get(vertex, set())

    def _build_neighbors_map(self, hop: int=-1) -> dict[Vertex, set[Vertex]]:
        # if hop == -1, return all reachable neighbors
        neighbor_map: dict[Vertex, set[Vertex]] = {}
        # for each vertex, calc its neighbors, store in neighbor_map
        # calculate for all vertices do not use neighbors function to avoid repeated calculation
        # since hypergraph is undirected, we can visit each vertex at most two, by using neighbors' neighbor
        # we can use a BFS-like approach to find all neighbors within the given hop distance
        distance_map: dict[tuple[Vertex, Vertex], int] = {}
        for vertex in self.vertices:
            visited: set[Vertex] = set()
            to_visit: set[Vertex] = {vertex}
            current_hop = hop
            while to_visit:
                current = to_visit.pop()
                visited.add(current)
                if current != vertex:
                    key = (vertex, current)
                    inv_key = (current, vertex)
                    if key not in distance_map or distance_map[key] > (hop - current_hop):
                        # Add distance_map entry
                        distance_map[key] = hop - current_hop
                        distance_map[inv_key] = hop - current_hop
                        if distance_map[key] <= hop or hop == -1:
                            if vertex not in neighbor_map:
                                neighbor_map[vertex] = set()
                            neighbor_map[vertex].add(current)
                            if current not in neighbor_map:
                                neighbor_map[current] = set()
                            neighbor_map[current].add(vertex)
                    
                    # solve other possible by `distance_map` and `neighbor_map`
                    if current in neighbor_map:
                        for neighbor in neighbor_map[current]:
                            if neighbor not in visited:
                                key2 = (vertex, neighbor)
                                inv_key2 = (neighbor, vertex)
                                if key2 not in distance_map or distance_map[key2] > (hop - current_hop + 1):
                                    distance_map[key2] = hop - current_hop + 1
                                    distance_map[inv_key2] = hop - current_hop + 1
                                    # Add neighbor to neighbor_map[vertex]
                                    if vertex not in neighbor_map:
                                        neighbor_map[vertex] = set()
                                    neighbor_map[vertex].add(neighbor)
                                    if neighbor not in neighbor_map:
                                        neighbor_map[neighbor] = set()
                                    neighbor_map[neighbor].add(vertex)
                                # if distance_map[(vertex, neighbor)] can be updated, add neighbor to to_visit
                                visited.add(neighbor)
                                if (hop - distance_map[key2]) > 0:
                                    for edge in self.contained_edges.get(neighbor, []):
                                        for next_neighbor in edge.vertices:
                                            if next_neighbor not in visited:
                                                to_visit.add(next_neighbor)
                if current_hop == 0:
                    continue
                for edge in self.contained_edges.get(current, []):
                    for neighbor in edge.vertices:
                        if neighbor not in visited:
                            to_visit.add(neighbor)
                if current_hop > 0:
                    current_hop -= 1
        for (from_vertex, to_vertex), dist in distance_map.items():
            if hop != -1 and dist > hop:
                continue
            if from_vertex not in neighbor_map:
                neighbor_map[from_vertex] = set()
            neighbor_map[from_vertex].add(to_vertex)
            if to_vertex not in neighbor_map:
                neighbor_map[to_vertex] = set()
            neighbor_map[to_vertex].add(from_vertex)
        return neighbor_map
    
    def paths(self, vertex1: Vertex, vertex2: Vertex) -> list[Path]:
        if (vertex1, vertex2) in self.path_map_cache:
            return self.path_map_cache[(vertex1, vertex2)]
        paths: list[Path] = []
        visited: set[Vertex] = set()
        to_visit: list[tuple[Vertex, list[Hyperedge]]] = [(vertex1, [])]
        while to_visit:
            current, current_path = to_visit.pop(0)
            visited.add(current)
            
            if (current, vertex2) in self.path_map_cache:
                cached_paths = self.path_map_cache[(current, vertex2)]
                for p in cached_paths:
                    full_path = current_path + p.hyperedges
                    paths.append(Path(full_path)) 
                continue
            
            if current == vertex2:
                paths.append(Path(current_path))
                continue
            for edge in self.contained_edges.get(current, []):
                for neighbor in edge.vertices:
                    if neighbor not in visited:
                        new_path = current_path + [edge]
                        to_visit.append((neighbor, new_path))

        self.path_map_cache[(vertex1, vertex2)] = paths
        return paths
