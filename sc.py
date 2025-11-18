from hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc, Path
from dependency import LocalDoc, Node, Pos
import numpy as np

from embedding import get_embedding_batch, cosine_similarity

from nli import get_nli_labels_batch

class TarjanLCA:
    def __init__(self, edges: list[tuple[Node, Node]], queries: list[tuple[Node, Node]]) -> None:
        # build adjacency list (undirected) and node set
        self.adj: dict[Node, list[Node]] = {}
        self.nodes: set[Node] = set()
        for a, b in edges:
            self.nodes.add(a)
            self.nodes.add(b)
            if a not in self.adj:
                self.adj[a] = []
            if b not in self.adj:
                self.adj[b] = []
            self.adj[a].append(b)
            self.adj[b].append(a)

        # store queries and build per-node query map
        self.queries = list(queries)
        self.query_map: dict[Node, list[tuple[Node, int]]] = {}
        for i, (u, v) in enumerate(self.queries):
            # queries are undirected for LCA
            if u not in self.query_map:
                self.query_map[u] = []
            if v not in self.query_map:
                self.query_map[v] = []
            self.query_map[u].append((v, i))
            self.query_map[v].append((u, i))

        # union-find parent and ancestor used by Tarjan's algorithm
        self.uf_parent: dict[Node, Node] = {}
        self.ancestor: dict[Node, Node] = {}
        self.visited: set[Node] = set()
        self.res: list[Node | None] = [None] * len(self.queries)

        # initialize union-find for nodes that appear in edges or queries
        for n in list(self.nodes):
            self.uf_parent[n] = n
            self.ancestor[n] = n

        # also include nodes that appear only in queries
        for u, v in self.queries:
            if u not in self.uf_parent:
                self.uf_parent[u] = u
                self.ancestor[u] = u
                self.nodes.add(u)
            if v not in self.uf_parent:
                self.uf_parent[v] = v
                self.ancestor[v] = v
                self.nodes.add(v)

        # run Tarjan on each component (forest support)
        for n in list(self.nodes):
            if n not in self.visited:
                self.tarjan(n, None)
        
    # union-find's find
    def find(self, x):
        # path compression
        if x not in self.uf_parent:
            self.uf_parent[x] = x
            return x
        if self.uf_parent[x] != x:
            self.uf_parent[x] = self.find(self.uf_parent[x])
        return self.uf_parent[x]
    
    # union-find's union
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        # attach ry under rx
        self.uf_parent[ry] = rx
    
    def tarjan(self, u, p):
        # standard Tarjan's offline LCA DFS
        # set ancestor of u to u
        self.ancestor[u] = u
        # visit children (neighbors excluding parent)
        for v in self.adj.get(u, []):
            if v == p:
                continue
            if v in self.visited:
                # already processed in this component
                continue
            self.tarjan(v, u)
            self.union(u, v)
            self.ancestor[self.find(u)] = u

        # mark u visited
        self.visited.add(u)

        # answer queries attached to u whenever the other node is already visited
        for other, qi in self.query_map.get(u, []):
            if other in self.visited:
                self.res[qi] = self.ancestor[self.find(other)]

    def lca(self) -> list[Node | None]:
        # return the results of all queries
        return self.res

class SemanticCluster:
    def __init__(self, hyperedges: list[Hyperedge], doc: LocalDoc, is_query: bool=True) -> None:
        self.hyperedges = hyperedges
        self.doc = doc
        self.vertices: list[Vertex] = []
        self.contained_hyperedges: dict[Vertex, list[Hyperedge]] = {}
        self.embedding: np.ndarray | None = None
        self.text_cache: str | None = None
        
        self.vertices_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
        self.node_paths_cache: dict[tuple[Node, Node], tuple[str, int]] = {}
        
        self.is_query = is_query
        
    @staticmethod
    def likely_nodes(nodes1: list[Vertex], nodes2: list[Vertex]) -> dict[Vertex, set[Vertex]]:
        # node is likely if NLI label is entailment or share same pos
        likely_nodes: dict[Vertex, set[Vertex]] = {}
        text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
        for node1 in nodes1:
            for node2 in nodes2:
                text_pair_to_node_pairs[(node1.text(), node2.text())] = (node1, node2)
        text_pairs = list(text_pair_to_node_pairs.keys())
        labels = get_nli_labels_batch(text_pairs)
        for i, text_pair in enumerate(text_pairs):
            node_pair = text_pair_to_node_pairs[text_pair]
            label = labels[i]
            node1, node2 = node_pair
            if label == "entailment" or node1.pos_same(node2):
                if node1 not in likely_nodes:
                    likely_nodes[node1] = set()
                likely_nodes[node1].add(node2)
        return likely_nodes
    
    
    def is_subset_of(self, other: 'SemanticCluster') -> bool:
        self_edge_set = set(self.hyperedges)
        other_edge_set = set(other.hyperedges)
        return self_edge_set.issubset(other_edge_set)
    
    def get_contained_hyperedges(self, vertex: Vertex) -> list[Hyperedge]:
        if vertex in self.contained_hyperedges:
            return self.contained_hyperedges[vertex]
        contained_edges: list[Hyperedge] = []
        for he in self.hyperedges:
            if vertex in he.vertices:
                contained_edges.append(he)
        self.contained_hyperedges[vertex] = contained_edges
        return contained_edges
    
    def get_vertices(self) -> list[Vertex]:
        if len(self.vertices) > 0:
            return self.vertices
        vertex_set: set[Vertex] = set()
        
        for he in self.hyperedges:
            for v in he.vertices:
                vertex_set.add(v)
        self.vertices = list(vertex_set)
        return self.vertices
    
    def get_paths_between_vertices(self, v1: Vertex, v2: Vertex) -> tuple[str, int]:
        key = (v1, v2)
        if key in self.vertices_paths:
            return self.vertices_paths[key]
        
        node_vertex: dict[Node, Vertex] = {}
        
        nodes_in_vertices: set[Node] = set()
        for he in self.hyperedges:
            for v in he.vertices:
                if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.ADJ):
                    continue
                nodes_in_vertices.add(he.current_node(v))
                node_vertex[he.current_node(v)] = v

        nodes_in_vertices_list = list(nodes_in_vertices)
        queries: list[tuple[Node, Node]] = []
        for i in range(len(nodes_in_vertices_list) - 1):
            for j in range(i + 1, len(nodes_in_vertices_list)):
                u = nodes_in_vertices_list[i]
                v = nodes_in_vertices_list[j]
                queries.append((u, v))
        
        edge_between_nodes: list[tuple[Node, Node]] = []
        for he in self.hyperedges:
            root = he.current_node(he.root)
            for i in range(1, len(he.vertices)):
                node = he.current_node(he.vertices[i])
                if node not in nodes_in_vertices:
                    continue
                current = node
                while current and current != root:
                    edge_between_nodes.append((root, current))
                    if current.head:
                        current = current.head
                    else:
                        break
                    
        lca_results = TarjanLCA(edge_between_nodes, queries).lca()
        
        lca_map: dict[tuple[Node, Node], Node] = {}
        for i, (u, v) in enumerate(queries):
            lca_node = lca_results[i]
            if lca_node:
                lca_map[(u, v)] = lca_node
        
        node_paths: dict[tuple[Vertex, Vertex], list[tuple[str, int]]] = {}
        
        for (u, v), k in lca_map.items():
            # collect path from u to k
            node_cnt = 1
            path_items: list[Node] = []
            current = u
            while current != k:
                if current in nodes_in_vertices:
                    node_cnt += 1
                path_items.append(current)
                assert current.head is not None, f"Node {current.text} has no head while tracing to LCA {k.text}"
                current = current.head
            path_items.append(k)
            # collect path from v to k
            rev_path_items: list[Node] = []
            current = v
            while current != k:
                if current in nodes_in_vertices:
                    node_cnt += 1
                rev_path_items.append(current)
                assert current.head is not None, f"Node {current.text} has no head while tracing to LCA {k.text}"
                current = current.head
            rev_path_items = rev_path_items[::-1]
            path_items.extend(rev_path_items)
            text = node_sequence_to_text(path_items)
            vertex_u = node_vertex[u]
            vertex_v = node_vertex[v]
            if (vertex_u, vertex_v) not in node_paths:
                node_paths[(vertex_u, vertex_v)] = []
            node_paths[(vertex_u, vertex_v)].append((text, node_cnt))
            if (vertex_v, vertex_u) not in node_paths:
                node_paths[(vertex_v, vertex_u)] = []
            node_paths[(vertex_v, vertex_u)].append((text, node_cnt))
            
        # select the shortest path
        for (vertex_u, vertex_v), paths in node_paths.items():
            paths = sorted(paths, key=lambda x: x[1])
            self.vertices_paths[(vertex_u, vertex_v)] = paths[0]
        
        return self.vertices_paths.setdefault(key, ("", 0))
        
    
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
            # print(f"Sentence: {sentence}")
            # print(f"Sentence by range: {sentence_by_range}")
            
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

def calc_embedding_for_cluster_batch(clusters: list[SemanticCluster]) -> None:
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
        

def get_semantic_cluster_pairs(query_hypergraph: Hypergraph, data_hypergraph: Hypergraph) -> list[tuple[SemanticCluster, SemanticCluster, float]]:
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
    
    # for node_q in likely_nodes:
    #     print(f"Query Node: {node_q.text()}, Likely Data Nodes: {[n.text() for n in likely_nodes[node_q]]}")
    
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
            # print(f"Query Vertex Pair: ({u.text()}, {v.text()}), Paths: {len(q_paths)} * {len(d_paths)} = {len(q_paths) * len(d_paths)}")
            q_clusters = [SemanticCluster(p.hyperedges, query_hypergraph.doc) for p in q_paths]
            d_clusters = [SemanticCluster(p.hyperedges, data_hypergraph.doc) for p in d_paths]
            calc_embedding_for_cluster_batch(q_clusters)
            calc_embedding_for_cluster_batch(d_clusters)
            
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

def node_sequence_to_text(nodes: list[Node]) -> str:
    start, end = nodes[0], nodes[-1]
    nodes = sorted(nodes, key=lambda n: n.index)
    texts = []
    for node in nodes:
        if node == start:
            texts.append("A")
        elif node == end:
            texts.append("B")
        elif node.pos in {Pos.ADV, Pos.ADJ, Pos.DET}:
            continue
        elif node.pos in {Pos.NOUN, Pos.PROPN, Pos.PRON}:
            texts.append("some")
        else:
            texts.append(node.text)
    return " ".join(texts)
        

def get_s_match(sc1: SemanticCluster, sc2: SemanticCluster) -> list[tuple[Vertex, Vertex]]:
    matches: list[tuple[Vertex, Vertex]] = []
    sc1_vertices = list(filter(lambda v: v.pos_equal(Pos.VERB) or v.pos_equal(Pos.ADJ), sc1.get_vertices()))
    sc2_vertices = list(filter(lambda v: v.pos_equal(Pos.VERB) or v.pos_equal(Pos.ADJ), sc2.get_vertices()))
    likely_nodes = SemanticCluster.likely_nodes(sc1_vertices, sc2_vertices)
    
    
    
    
    
    return matches