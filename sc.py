import itertools
from os import path
from hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc, Path
from dependency import LocalDoc, Node, Pos
import numpy as np

from embedding import get_embedding_batch, cosine_similarity, get_similarity_batch, get_similarity

from nli import get_nli_label, get_nli_labels_batch

class TarjanLCA:
    def __init__(self, edges: list[tuple[Node, Node]], queries: list[tuple[Node, Node]]) -> None:
        # build adjacency list (directed) and node set
        self.adj: dict[Node, list[Node]] = {}
        self.nodes: set[Node] = set()
        
        # 统计入度，用于寻找有向图/树的根节点
        in_degree: dict[Node, int] = {}

        for a, b in edges:
            self.nodes.add(a)
            self.nodes.add(b)
            if a not in self.adj:
                self.adj[a] = []
            self.adj[a].append(b)
            
            # 初始化入度
            if a not in in_degree: in_degree[a] = 0
            if b not in in_degree: in_degree[b] = 0
            in_degree[b] += 1

        # store queries and build per-node query map
        self.queries = list(queries)
        self.query_map: dict[Node, list[tuple[Node, int]]] = {}
        
        for i, (u, v) in enumerate(self.queries):
            self.nodes.add(u)
            self.nodes.add(v)
            if u not in in_degree: in_degree[u] = 0
            if v not in in_degree: in_degree[v] = 0

            # 建立双向映射
            if u not in self.query_map: self.query_map[u] = []
            if v not in self.query_map: self.query_map[v] = []
            
            self.query_map[u].append((v, i))
            if u != v:
                self.query_map[v].append((u, i))

        # union-find parent and ancestor used by Tarjan's algorithm
        self.uf_parent: dict[Node, Node] = {}
        self.ancestor: dict[Node, Node] = {}
        self.visited: set[Node] = set()
        self.res: list[Node | None] = [None] * len(self.queries)

        # [新增逻辑] 用于记录节点属于哪棵树（哪个连通分量）
        self.node_roots: dict[Node, Node] = {}

        # initialize union-find for all nodes
        for n in list(self.nodes):
            self.uf_parent[n] = n
            self.ancestor[n] = n

        # run Tarjan on each component (forest support)
        # 优先从根节点（入度为0）开始 DFS
        sorted_nodes = sorted(list(self.nodes), key=lambda n: in_degree.get(n, 0))
        
        for n in sorted_nodes:
            if n not in self.visited:
                # [修改逻辑] 传入当前分量的根节点 n 作为 root_id
                self.tarjan(n, None, n)
        
    # union-find's find
    def find(self, x):
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
        self.uf_parent[ry] = rx
    
    # [修改接口] 增加 root_id 参数，标记当前递归属于哪棵树
    def tarjan(self, u, p, root_id):
        # [新增逻辑] 记录当前节点所属的树根
        self.node_roots[u] = root_id

        self.ancestor[u] = u 
        
        for v in self.adj.get(u, []):
            if v == p: 
                continue
            if v in self.visited:
                continue
            
            # [修改逻辑] 递归传递 root_id
            self.tarjan(v, u, root_id)
            self.union(u, v)
            self.ancestor[self.find(u)] = u

        self.visited.add(u)

        for other, qi in self.query_map.get(u, []):
            # [修复核心] 只有当 other 也被访问过，且 other 属于同一棵树（同一个 root_id）时，才计算 LCA
            # 如果属于不同的树，说明不连通，LCA 保持为 None
            if other in self.visited:
                if self.node_roots.get(other) == root_id:
                    self.res[qi] = self.ancestor[self.find(other)]

    def lca(self) -> list[Node | None]:
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
            if label == "entailment" or (label == "neutral" and node1.is_domain(node2)):
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
        id_set = set()
        
        for he in self.hyperedges:
            for v in he.vertices:
                if v.id in id_set:
                    continue
                id_set.add(v.id)
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
                if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
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
        saved_nodes: set[Node] = set()
        for he in self.hyperedges:
            root = he.current_node(he.root)
            for i in range(1, len(he.vertices)):
                node = he.current_node(he.vertices[i])
                edge_between_nodes.append((root, node))
                saved_nodes.add(node) # HINT
            head = root.head
            current = root
            while head:
                edge_between_nodes.append((head, current))
                if head in saved_nodes:
                    break
                current = head
                head = head.head
            saved_nodes.add(root)
            
        # for (u, v) in edge_between_nodes:
        #     print(f"Edge: '{u.text} ({u.index})' -> '{v.text} ({v.index})'")
                    
        lca_results = TarjanLCA(edge_between_nodes, queries).lca()
        
        
        lca_map: dict[tuple[Node, Node], Node] = {}
        for i, (u, v) in enumerate(queries):
            lca_node = lca_results[i]
            if lca_node:
                # print(f"LCA of '{u.text} ({u.index})' and '{v.text} ({v.index})' is '{lca_node.text} ({lca_node.index})'")
                lca_map[(u, v)] = lca_node
        
        node_paths: dict[tuple[Vertex, Vertex], list[tuple[str, int]]] = {}
        
        for (u, v), k in lca_map.items():
            # collect path from u to k
            # print(f"Collecting path between '{u.text} ({u.index})' and '{v.text} ({v.index})', LCA is '{k.text} ({k.index})'")
            node_cnt = 1
            path_items: list[Node] = []
            current = u
            current_trace: list[str] = []
            while current != k:
                current_trace.append(current.text)
                if current in nodes_in_vertices:
                    node_cnt += 1
                path_items.append(current)
                assert current.head is not None, f"Node '{current.text}' has no head while tracing to LCA '{k.text}', current trace: {' -> '.join(current_trace)}"
                current = current.head
                
            path_items.append(k)
            # collect path from v to k
            rev_path_items: list[Node] = []
            current = v
            current_trace: list[str] = []
            while current != k:
                current_trace.append(current.text)
                if current in nodes_in_vertices:
                    node_cnt += 1
                rev_path_items.append(current)
                assert current.head is not None, f"Node '{current.text}' has no head while tracing to LCA '{k.text}', current trace: {' -> '.join(current_trace)}"
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
                node_text = Vertex.resolved_text(node)
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
        if label == "entailment" or (label == "neutral" and node_q.is_domain(node_d)):
            if node_q not in likely_nodes:
                likely_nodes[node_q] = set()
            likely_nodes[node_q].add(node_d)
    
    # for node_q in likely_nodes:
        # print(f"Query Node: {node_q.text()}, Likely Data Nodes: {[n.text() for n in likely_nodes[node_q]]}")
    
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
            # print(f"Query Vertex Pair: ({u.text()}, {v.text()}), Top K: {len(top_k)}")
            # for qc, dc, score in top_k:
                # print(f"Query Cluster Text: {qc.text()}, Data Cluster Text: {dc.text()}, Score: {score:.4f}")
                # print("-----")
    
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
        

def _legal_path(s1: str, cnt1: int, s2: str, cnt2: int) -> bool:
    # legal if s1 == s2 or (cnt1 > 2 and cnt2 > 2)
    if get_nli_label(s1, s2) != "contradiction":
        return True
    return False

def _legal_vertices(v1: Vertex, v2: Vertex) -> bool:
    label = get_nli_label(v1.text(), v2.text())
    if label == "entailment" or v1.is_domain(v2):
        return True
    return False

def _path_score(s1: str, cnt1: int, s2: str, cnt2: int, path_score_cache: dict[tuple[str, str], float]) -> float:
    if (s1, s2) in path_score_cache:
        sim = path_score_cache[(s1, s2)]
    else:
        sim = get_similarity(s1, s2)
        path_score_cache[(s1, s2)] = sim
    return sim / (cnt1 + cnt2)

def get_d_match(sc1: SemanticCluster, sc2: SemanticCluster, score_threshold: float=0.0) -> list[tuple[Vertex, Vertex, float]]:
    matches: list[tuple[Vertex, Vertex]] = []
    sc1_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc1.get_vertices()))
    sc2_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc2.get_vertices()))
    
    sc1_edges: list[tuple[Vertex, Vertex]] = []
    for he in sc1.hyperedges:
        for i in range(len(he.vertices) - 1):
            for j in range(i + 1, len(he.vertices)):
                if he.vertices[i].pos_equal(Pos.VERB) or he.vertices[i].pos_equal(Pos.AUX):
                    continue
                if he.vertices[j].pos_equal(Pos.VERB) or he.vertices[j].pos_equal(Pos.AUX):
                    continue
                if he.is_sub_vertex(he.vertices[i], he.vertices[j]):
                    sc1_edges.append((he.vertices[i], he.vertices[j]))
                else:
                    sc1_edges.append((he.vertices[j], he.vertices[i]))
                    
    index_map: dict[Vertex, int] = {}
    for e in sc1.hyperedges:
        for v in e.vertices:
            if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                continue
            if v not in index_map:
                index_map[v] = e.current_node(v).index
    
    sc1_pairs : list[tuple[Vertex, Vertex]] = []
    # all (u, v) in sc1_edges are in sc1_pairs, and if (u, k), (k, v) in sc1_edges, then (u, v) is also in sc1_pairs
    # calculate then recursively
    added = True
    for u, v in sc1_edges:
        sc1_pairs.append((u, v))
    while added:
        added = False
        current_pairs = sc1_pairs.copy()
        for u1, v1 in current_pairs:
            for u2, v2 in current_pairs:
                if v1 == u2:
                    new_pair = (u1, v2)
                    if new_pair not in sc1_pairs:
                        sc1_pairs.append(new_pair)
                        added = True
    
    sc1_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    sc1_is_textual_inverse: set[tuple[Vertex, Vertex]] = set()
    for u, v in sc1_pairs:
        s, cnt = sc1.get_paths_between_vertices(u, v)
        sc1_paths[(u, v)] = (s, cnt)
        if index_map[u] > index_map[v]:
            sc1_is_textual_inverse.add((u, v))
    
    likely_nodes = SemanticCluster.likely_nodes(sc1_vertices, sc2_vertices)
    
    sc2_pairs: list[tuple[Vertex, Vertex]] = []
    sc2_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    
    for u, u_prime in sc1_pairs:
        for v, v_prime in itertools.product(likely_nodes.get(u, set()), likely_nodes.get(u_prime, set())):
            s1, cnt1 = sc1_paths[(u, u_prime)]
            if (v, v_prime) in sc2_paths:
                s2, cnt2 = sc2_paths[(v, v_prime)]
            else:
                s2, cnt2 = sc2.get_paths_between_vertices(v, v_prime)
            if _legal_path(s1, cnt1, s2, cnt2):
                sc2_pairs.append((v, v_prime))
                sc2_paths[(v, v_prime)] = (s2, cnt2)
    
    match_scores: dict[tuple[Vertex, Vertex], float] = {}
    
    for u, v in itertools.product(sc1_vertices, sc2_vertices):
        if _legal_vertices(u, v):
            matches.append((u, v))
    
    in_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc1_pairs:
        if v not in in_paths_of_sc1:
            in_paths_of_sc1[v] = []
        in_paths_of_sc1[v].append(sc1_paths[(u, v)])
        if u not in out_paths_of_sc1:
            out_paths_of_sc1[u] = []
        out_paths_of_sc1[u].append(sc1_paths[(u, v)])
        
    in_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc2_pairs:
        if v not in in_paths_of_sc2:
            in_paths_of_sc2[v] = []
        in_paths_of_sc2[v].append(sc2_paths[(u, v)])
        if u not in out_paths_of_sc2:
            out_paths_of_sc2[u] = []
        out_paths_of_sc2[u].append(sc2_paths[(u, v)])
    
    path_score_cache: dict[tuple[str, str], float] = {}
    path_pair_need_to_calc: set[tuple[str, str]] = set()
    for u, v in matches:
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                key = (s1, s2)
                path_pair_need_to_calc.add(key)
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                key = (s1, s2)
                path_pair_need_to_calc.add(key)
    
    path_list_1: list[str] = []
    path_list_2: list[str] = []
    for s1, s2 in path_pair_need_to_calc:
        path_list_1.append(s1)
        path_list_2.append(s2)
    similarities = get_similarity_batch(path_list_1, path_list_2)
    for i, (s1, s2) in enumerate(path_pair_need_to_calc):
        path_score_cache[(s1, s2)] = similarities[i]
    
    for u, v in matches:
        in_score = 0.0
        out_score = 0.0
        in_cnt = 0
        out_cnt = 0
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                in_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                in_cnt += 1
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                out_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                out_cnt += 1
        # in_score are the average by all in_path pairs
        if in_cnt > 0:
            in_score /= in_cnt
        if out_cnt > 0:
            out_score /= out_cnt
        match_scores[(u, v)] = in_score + out_score
        
    # filter by score_threshold
    matches = list(filter(lambda pair: match_scores.get(pair, 0.0) >= score_threshold, matches))
    
    # delete the matches that if (u, v1) and (u, v2) in matches and v1 != v2, keep only the one with highest score
    final_matches: list[tuple[Vertex, Vertex, float]] = []
    matches_by_u: dict[Vertex, list[tuple[Vertex, float]]]  = {}
    for u, v in matches:
        score = match_scores.get((u, v), 0.0)
        if u not in matches_by_u:
            matches_by_u[u] = []
        matches_by_u[u].append((v, score))
    for u, v_scores in matches_by_u.items():
        v_scores = sorted(v_scores, key=lambda x: x[1], reverse=True)
        best_v, best_score = v_scores[0]
        final_matches.append((u, best_v, best_score))
    
    return final_matches