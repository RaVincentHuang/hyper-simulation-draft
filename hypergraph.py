from math import cos
from scipy import cluster
from dependency import LocalDoc, Node, Pos, Relationship

import numpy as np
from embedding import get_embedding_batch, cosine_similarity

from nli import get_nli_labels_batch

# from index import build_faiss_index, search_faiss

class Vertex:
    def __init__(self, id: int, nodes: list[Node]):
        self.id = id
        self.nodes = nodes
        
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

class SemanticCluster:
    def __init__(self, hyperedges: list[Hyperedge], doc: LocalDoc) -> None:
        self.hyperedges = hyperedges
        self.doc = doc
        self.embedding: np.ndarray | None = None
        self.text_cache: str | None = None
        
    def is_subset_of(self, other: 'SemanticCluster') -> bool:
        self_edge_set = set(self.hyperedges)
        other_edge_set = set(other.hyperedges)
        return self_edge_set.issubset(other_edge_set)
    
    def __hash__(self) -> int:
        return hash(frozenset(self.hyperedges))
    
    def text(self) -> str:
        if self.text_cache is not None:
            return self.text_cache
        
        if not self.hyperedges:
            return ""
        
        # Calc all the roots
        root_ancestors = {e.current_node(e.root): e.current_node(e.root) for e in self.hyperedges}
        # print(f"Nodes: {[e.current_node(e.root).text for e in self.hyperedges]}")
        for e in self.hyperedges:
            root = e.current_node(e.root)
            node = root
            ancestors = []
            while node.head:
                ancestors.append(node)
                # print(f" {root.text}'s ancestor node: {node.head.text}")
                if node.head in root_ancestors:
                    root_ancestors[root] = root_ancestors[node.head]
                    break
                node = node.head
        
        root_to_nodes: dict[Node, set[Node]] = {}
        for e in self.hyperedges:
            root = e.current_node(e.root)
            root_of_root = root_ancestors[root]
            if root_of_root not in root_to_nodes:
                root_to_nodes[root_of_root] = set()
            for vertex in e.vertices:
                node = e.current_node(vertex)
                root_to_nodes[root_of_root].add(node)
                
        sub_cluster_roots: set[Node] = set()
        for root, nodes in root_to_nodes.items():
            sub_cluster_roots.add(root_ancestors[root])
        
        sub_clusters = sorted(list(sub_cluster_roots), key=lambda r: r.index)
        
        texts = []
        
        # print(f"Sub-clusters count: {len(sub_clusters)}")
        
        for root in sub_clusters:
            nodes = list(root_to_nodes[root])
            start = min(node.index for node in nodes)
            end = max(node.index for node in nodes) + 1
            sentence_by_range = str(self.doc[start:end])
            sentence = str(root.sentence)
            print(f"Sentence: {sentence}")
            print(f"Sentence by range: {sentence_by_range}")
            
            def calc_prefix_suffix(sentence_by_range, sentence):
                start = sentence.find(sentence_by_range)
                if start != -1:
                    prefix = sentence[:start].strip()
                    suffix = sentence[start + len(sentence_by_range):].strip()
                else:
                    prefix = ""
                    suffix = ""
                return prefix, suffix
        
            prefix, suffix = calc_prefix_suffix(sentence_by_range, sentence)
            
            replacement = []
            for node in nodes:
                if node == root:
                    continue
                if node.pos == Pos.PRON and node.pronoun_antecedent:
                    node_text = node.pronoun_antecedent.text
                else:
                    node_text = node.text
                replacement.append((node.sentence, node_text))
            
            replacement.append((prefix, ""))
            replacement.append((suffix, ""))
            
            for old, new in replacement:
                sentence = sentence.replace(old, new)
            
            texts.append(sentence.strip())
        
        # for text in texts:
        #     print(f" Sub-cluster text: {text}")
        
        text = " ".join(texts).strip()
        
        self.text_cache = text
        return text

def calc_embbeding_for_cluster_batch(clusters: list[SemanticCluster]) -> None:
    texts = [sc.text() for sc in clusters]
    embeddings = get_embedding_batch(texts)
    for i, sc in enumerate(clusters):
        sc.embedding = np.array(embeddings[i])


def path_clean(paths: list[Path]) -> list[Path]:
    # remove paths that share same hyperedges
    # in a path, hyperedges are unique, if not, keep only one
    
    # 1. for each path, remove duplicate hyperedges
    cleaned_paths: list[Path] = []
    for path in paths:
        seen_hyperedges: set[int] = set()
        unique_hyperedges: list[Hyperedge] = []
        for he in path.hyperedges:
            he_id = id(he)
            if he_id not in seen_hyperedges:
                seen_hyperedges.add(he_id)
                unique_hyperedges.append(he)
        cleaned_paths.append(Path(unique_hyperedges))
    
    unique_paths: list[Path] = []
    seen_hyperedge_sets: set[frozenset[int]] = set()
    for path in cleaned_paths:
        hyperedge_ids = frozenset(id(e) for e in path.hyperedges)
        if hyperedge_ids not in seen_hyperedge_sets:
            seen_hyperedge_sets.add(hyperedge_ids)
            unique_paths.append(path)
    
    return unique_paths
    

def clean_semantic_cluster_pairs(pairs: list[tuple[SemanticCluster, SemanticCluster, float]]) -> list[tuple[SemanticCluster, SemanticCluster, float]]:
    # remove pairs where one cluster (qc, dc, score) if there exists another pair (qc', dc', score')
    # and score <= score' and (qc is subset of qc' or dc is subset of dc') 
    cleaned_pairs: list[tuple[SemanticCluster, SemanticCluster, float]] = []
    for i, (qc, dc, score) in enumerate(pairs):
        is_subset = False
        for j, (qc2, dc2, score2) in enumerate(pairs):
            if i == j:
                continue
            if score <= score2:
                if qc2.is_subset_of(qc) or dc2.is_subset_of(dc):
                    is_subset = True
                    break
        if not is_subset:
            cleaned_pairs.append((qc, dc, score))
    return cleaned_pairs
        
    
    

def get_semantic_cluster_pairs(query_hypergraph: Hypergraph, data_hypergraph: Hypergraph):
    single_cluster_q: list[SemanticCluster] = []
    for e in query_hypergraph.hyperedges:
        single_cluster_q.append(SemanticCluster([e], query_hypergraph.doc))
        
    # calc the embeddings by `get_embedding_batch` of single_cluster_q
    texts_q = [sc.text() for sc in single_cluster_q]
    embeddings_q = get_embedding_batch(texts_q)
    for i, sc in enumerate(single_cluster_q):
        sc.embedding = np.array(embeddings_q[i])
    
    single_cluster_d: list[SemanticCluster] = []
    for e in data_hypergraph.hyperedges:
        single_cluster_d.append(SemanticCluster([e], data_hypergraph.doc))
    
    texts_d = [sc.text() for sc in single_cluster_d]
    embeddings_d = get_embedding_batch(texts_d)
    for i, sc in enumerate(single_cluster_d):
        sc.embedding = np.array(embeddings_d[i])
    
    N_STEP = -1
    
    text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
    for node_q in query_hypergraph.vertices:
        for node_d in data_hypergraph.vertices:
            text_pair_to_node_pairs[(node_q.text(), node_d.text())] = (node_q, node_d)
    
    # send text_pair_to_node_pairs's keys (tuple[str, str]) to get_nli_labels_batch, get the labels for the value (tuple[Node, Node])
    text_pairs = list(text_pair_to_node_pairs.keys())
    labels = get_nli_labels_batch(text_pairs)
    node_pair_to_label: dict[tuple[Vertex, Vertex], str] = {}
    for i, text_pair in enumerate(text_pairs):
        node_pair = text_pair_to_node_pairs[text_pair]
        node_pair_to_label[node_pair] = labels[i]
    
    likely_nodes: dict[Vertex, set[Vertex]] = {}
    for (node_q, node_d), label in node_pair_to_label.items():
        if label != "contradiction":
            if node_q not in likely_nodes:
                likely_nodes[node_q] = set()
            likely_nodes[node_q].add(node_d)
    
    for node_q in likely_nodes:
        print(f"Query Node: {node_q.text()}, Likely Data Nodes: {[n.text() for n in likely_nodes[node_q]]}")
    
    cluster_pairs: set[tuple[SemanticCluster, SemanticCluster, float]] = set()
    for u in query_hypergraph.vertices:
        for v in query_hypergraph.neighbors(u, N_STEP):
            q_paths = query_hypergraph.paths(u, v)
            u_prime_candidates = likely_nodes.get(u, set())
            v_prime_candidates = likely_nodes.get(v, set())
            d_paths = []
            for u_prime in u_prime_candidates:
                for v_prime in v_prime_candidates:
                    d_paths.extend(data_hypergraph.paths(u_prime, v_prime))
            q_paths = path_clean(q_paths)
            d_paths = path_clean(d_paths)
            print(f"Query Vertex Pair: ({u.text()}, {v.text()}), Paths: {len(q_paths)} * {len(d_paths)} = {len(q_paths) * len(d_paths)}")
            q_clusters = [SemanticCluster(p.hyperedges, query_hypergraph.doc) for p in q_paths]
            d_clusters = [SemanticCluster(p.hyperedges, data_hypergraph.doc) for p in d_paths]
            calc_embbeding_for_cluster_batch(q_clusters)
            calc_embbeding_for_cluster_batch(d_clusters)
            
            k = 10
            top_k = []
            for qc in q_clusters:
                if qc.embedding is None:
                    continue
                for dc in d_clusters:
                    if dc.embedding is None:
                        continue
                    score = cosine_similarity(qc.embedding, dc.embedding)
                    top_k.append((qc, dc, score))
            top_k = clean_semantic_cluster_pairs(top_k)
            # top_k = sorted(top_k, key=lambda x: x[2], reverse=True)[:k]
            # cluster_pairs.(top_k)
            for triplet in top_k:
                cluster_pairs.add(triplet)
    
    # remove all same pairs
    ans_pairs = []
    seen_pairs: set[tuple[frozenset[int], frozenset[int]]] = set()
    for qc, dc, score in cluster_pairs:
        qc_id_set = frozenset(id(e) for e in qc.hyperedges)
        dc_id_set = frozenset(id(e) for e in dc.hyperedges)
        pair_key = (qc_id_set, dc_id_set)
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            ans_pairs.append((qc, dc, score))

    return ans_pairs
            
    
    

# A Lattice of Hyperedges, where each Hyperedge is connected to others via shared Vertices.
# Form each Hyperedge as a node, end at hyperedges connected component
# class Lattice:
#     def __init__(self, hyperedges: list[Hyperedge]):
#         self.hyperedges = hyperedges
#         self.adjacency: dict[Hyperedge, set[Hyperedge]] = {}
#         for i, he1 in enumerate(hyperedges):
#             for j in range(i + 1, len(hyperedges)):
#                 he2 = hyperedges[j]
#                 if set(he1.vertices) & set(he2.vertices):
#                     if he1 not in self.adjacency:
#                         self.adjacency[he1] = set()
#                     if he2 not in self.adjacency:
#                         self.adjacency[he2] = set()
#                     self.adjacency[he1].add(he2)
#                     self.adjacency[he2].add(he1)
        
    
#     def connected_components(self) -> list[set[Hyperedge]]:
#         visited: set[Hyperedge] = set()
#         components: list[set[Hyperedge]] = []
#         for hyperedge in self.hyperedges:
#             if hyperedge in visited:
#                 continue
#             component: set[Hyperedge] = set()
#             to_visit: set[Hyperedge] = {hyperedge}
#             while to_visit:
#                 current = to_visit.pop()
#                 visited.add(current)
#                 component.add(current)
#                 for neighbor in self.adjacency.get(current, []):
#                     if neighbor not in visited:
#                         to_visit.add(neighbor)
#             components.append(component)
#         return components