from platform import node
from dependency import LocalDoc, Node, Pos, Entity, Relationship, Dep, Tag
import itertools
import os
from datetime import datetime

import index

# Debug 日志管理器
class DebugLogger:
    _instance = None
    _file = None
    _log_dir = "debug_logs"
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def init(cls, filename: str = None):
        """初始化日志文件，filename 为 None 时使用时间戳命名"""
        inst = cls.get_instance()
        if inst._file:
            inst._file.close()
        
        os.makedirs(cls._log_dir, exist_ok=True)
        if filename is None:
            filename = f"is_domain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(cls._log_dir, filename)
        inst._file = open(filepath, 'w', encoding='utf-8')
        inst.log(f"Debug log started at {datetime.now()}")
        return filepath
    
    @classmethod
    def log(cls, msg: str):
        inst = cls.get_instance()
        if inst._file:
            inst._file.write(msg + '\n')
            inst._file.flush()
    
    @classmethod
    def close(cls):
        inst = cls.get_instance()
        if inst._file:
            inst._file.close()
            inst._file = None

class Vertex:
    def __init__(self, id: int, nodes: list[Node]):
        self.id = id
        self.nodes = nodes
        self.poses: list[Pos] = [n.pos for n in nodes]
        self.ents: list[Entity] = [n.ent for n in nodes]
        # Deduplication for poses and ents
        self.poses = list(set(self.poses))
        self.ents = list(set(self.ents))
        
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vertex):
            return False
        return self.id == other.id
        
    def pos_equal(self, pos: Pos) -> bool:
        if not len(self.poses):
            return False
        return any(p == pos for p in self.poses)
    
    def pos_range(self, pos: Pos):
        # we assume that if two pos in a set, the are in a range
        # 1. AUX, VERB
        # 2. NOUN, PROPN, PRON
        # 3. CCONJ, SCONJ
        # 4. PUNCT, SYM, X
        if pos in {Pos.VERB, Pos.AUX}:
            return any(p in {Pos.VERB, Pos.AUX} for p in self.poses)
        elif pos in {Pos.NOUN, Pos.PROPN, Pos.PRON}:
            return any(p in {Pos.NOUN, Pos.PROPN, Pos.PRON} for p in self.poses)
        elif pos in {Pos.CCONJ, Pos.SCONJ}:
            return any(p in {Pos.CCONJ, Pos.SCONJ} for p in self.poses)
        elif pos in {Pos.PUNCT, Pos.SYM, Pos.X}:
            return any(p in {Pos.PUNCT, Pos.SYM, Pos.X} for p in self.poses)
        else:
            return any(p == pos for p in self.poses)
    
    def ent_equal(self, ent: Entity) -> bool:
        if not len(self.ents):
            return False
        return any(n.ent == ent for n in self.nodes)
    
    def ent_range(self, ent: Entity) -> bool:
        # Same as pos_range
        # 1. Person: PERSON, NORP
        # 2. Location: NORP, GPE, LOC, FAC, ORG
        # 3. Date/Time: DATE, TIME
        # 4. Object: PRODUCT, WORK_OF_ART
        # 5. Number: MONEY, PERCENT, QUANTITY, CARDINAL
        # 6. Fact: EVENT, LAW, LANGUAGE
        if ent in {Entity.PERSON, Entity.NORP}:
            return any(n.ent in {Entity.PERSON, Entity.NORP} for n in self.nodes)
        elif ent in {Entity.GPE, Entity.LOC, Entity.FAC, Entity.ORG, Entity.NORP}:
            return any(n.ent in {Entity.GPE, Entity.LOC, Entity.FAC, Entity.ORG, Entity.NORP} for n in self.nodes)
        elif ent in {Entity.DATE, Entity.TIME}:
            return any(n.ent in {Entity.DATE, Entity.TIME} for n in self.nodes)
        elif ent in {Entity.PRODUCT, Entity.WORK_OF_ART}:
            return any(n.ent in {Entity.PRODUCT, Entity.WORK_OF_ART} for n in self.nodes)
        elif ent in {Entity.MONEY, Entity.PERCENT, Entity.QUANTITY, Entity.CARDINAL}:
            return any(n.ent in {Entity.MONEY, Entity.PERCENT, Entity.QUANTITY, Entity.CARDINAL} for n in self.nodes)
        elif ent in {Entity.EVENT, Entity.LAW, Entity.LANGUAGE}:
            return any(n.ent in {Entity.EVENT, Entity.LAW, Entity.LANGUAGE} for n in self.nodes)
        else:
            return any(n.ent == ent for n in self.nodes)
    
    def ent_same(self, other: 'Vertex') -> bool:
        if not self.ents or not other.ents:
            return False
        return any(e1 == e2 for (e1, e2) in itertools.product(self.ents, other.ents))
        
    def pos_same(self, other: 'Vertex') -> bool:
        if not self.poses or not other.poses:
            return False
        return any(pos1 == pos2 for (pos1, pos2) in itertools.product(self.poses, other.poses))
        
    def is_domain(self, other: 'Vertex', debug: bool = False) -> bool:
        # 获取文本用于 debug
        self_text = ', '.join([n.text for n in self.nodes])
        other_text = ', '.join([n.text for n in other.nodes])
        
        def _log(msg):
            if debug:
                DebugLogger.log(msg)
        
        _log(f"\n{'='*60}")
        _log(f"[is_domain] 比较: '{self_text}' vs '{other_text}'")
        _log(f"  Self:  ID={self.id}, POS={[p.name for p in self.poses]}, ENT={[e.name for e in self.ents]}")
        _log(f"  Other: ID={other.id}, POS={[p.name for p in other.poses]}, ENT={[e.name for e in other.ents]}")

        # 1. NER 实体匹配 第一步排除
        self_has_ent = any(e != Entity.NOT_ENTITY for e in self.ents)
        other_has_ent = any(e != Entity.NOT_ENTITY for e in other.ents)
        
        if self_has_ent and other_has_ent:
            matched = any(self.ent_range(e) for e in other.ents if e != Entity.NOT_ENTITY)
            self_ents = [e.name for e in self.ents if e != Entity.NOT_ENTITY]
            other_ents = [e.name for e in other.ents if e != Entity.NOT_ENTITY]
            _log(f"  [1.NER] self={self_ents}, other={other_ents} → {'✓ 匹配' if matched else '✗ 不匹配'}")
            return matched
        
        if self_has_ent != other_has_ent:
            _log(f"  [1.NER] 一方有实体一方无 → ✗ False")
            return False

        # 2. WordNet 第二步排除不一致的
        wn_result = self._wordnet_domain_match(other, debug=debug)
        if wn_result is False:
            _log(f"  [2.WordNet] 最终结果 → {'✗'}")
            return wn_result

        # 3. Wikidata
        wd_result = self._wikidata_domain_match(other, debug=debug)
        if wd_result is not None:
            _log(f"  [3.Wikidata] 最终结果 → {'✓' if wd_result else '✗'}")
            return wd_result

        # 4. POS fallback
        pos_matched = any(self.pos_range(p) for p in other.poses)
        _log(f"  [4.POS] self={[p.name for p in self.poses]}, other={[p.name for p in other.poses]} → {'✓' if pos_matched else '✗'}")
        return pos_matched
    
    def _wordnet_domain_match(self, other: 'Vertex', debug: bool = False) -> bool | None:
        def _log(msg):
            if debug:
                DebugLogger.log(msg)
        
        self_abs = {n.wn_abstraction for n in self.nodes if getattr(n, 'wn_abstraction', None)}
        other_abs = {n.wn_abstraction for n in other.nodes if getattr(n, 'wn_abstraction', None)}
        
        self_text = ', '.join([f"'{n.text}'→{getattr(n, 'wn_abstraction', None)}" for n in self.nodes])
        other_text = ', '.join([f"'{n.text}'→{getattr(n, 'wn_abstraction', None)}" for n in other.nodes])
        _log(f"  [2.WordNet] 抽象类型:")
        _log(f"    Self:  {self_text}")
        _log(f"    Other: {other_text}")
        
        if self_abs and other_abs:
            common = self_abs & other_abs
            _log(f"    交集: {common if common else '(空)'} → {'✓' if common else '✗'}")
            return bool(common)

        self_hyp = {h for n in self.nodes for h in getattr(n, 'wn_hypernym_path', [])}
        other_hyp = {h for n in other.nodes for h in getattr(n, 'wn_hypernym_path', [])}

        if not self_hyp or not other_hyp:
            _log(f"    上位词路径缺失 (self={len(self_hyp)}, other={len(other_hyp)}) → 跳过")
            return None

        top_level = {'entity.n.01', 'abstraction.n.06', 'physical_entity.n.01'}
        common = (self_hyp & other_hyp) - top_level
        result = len(common) > 0
        _log(f"    上位词交集(排除顶层): {list(common)[:5]}{'...' if len(common) > 5 else ''} → {'✓' if result else '✗'}")
        return result

    def _wikidata_domain_match(self, other: 'Vertex', debug: bool = False) -> bool | None:
        """基于 Wikidata 标签判断语义领域（运行时查询），返回 None 表示无法判断"""
        from linking import WikidataTagger
        
        def _log(msg):
            if debug:
                DebugLogger.log(msg)

        self_pairs = [(n.text, n.sentence) for n in self.nodes if n.pos in {Pos.NOUN, Pos.PROPN}]
        other_pairs = [(n.text, n.sentence) for n in other.nodes if n.pos in {Pos.NOUN, Pos.PROPN}]
        
        if not self_pairs or not other_pairs:
            _log(f"  [3.Wikidata] 节点为空 → 跳过")
            return None

        _log(f"  [3.Wikidata] 查询中...")
        _log(f"    Self 查询: {[p[0] for p in self_pairs]}")
        _log(f"    Other 查询: {[p[0] for p in other_pairs]}")

        tagger = WikidataTagger()
        all_pairs = self_pairs + other_pairs
        all_results = tagger.batch_process(all_pairs)

        self_results = all_results[:len(self_pairs)]
        other_results = all_results[len(self_pairs):]

        self_wd_values = set()
        for i, res in enumerate(self_results):
            for v in res.values():
                self_wd_values.update(v.lower().split('; '))
            if res:
                _log(f"    '{self_pairs[i][0]}' → {dict(res)}")

        other_wd_values = set()
        for i, res in enumerate(other_results):
            for v in res.values():
                other_wd_values.update(v.lower().split('; '))
            if res:
                _log(f"    '{other_pairs[i][0]}' → {dict(res)}")

        if not self_wd_values or not other_wd_values:
            _log(f"    标签为空 (self={len(self_wd_values)}, other={len(other_wd_values)}) → 跳过")
            return None

        common_tags = self_wd_values & other_wd_values
        result = bool(common_tags)

        if result:
            _log(f"    共同标签: {common_tags} → ✓")
        else:
            _log(f"    无共同标签:")
            _log(f"      Self:  {self_wd_values}")
            _log(f"      Other: {other_wd_values}")

        return result

    @staticmethod
    def resolved_text(node: 'Node') -> str:
        """Get resolved text for a node, handling coreference and pronoun antecedents."""
        if node.resolved_text:
            return node.resolved_text
        # If node has a coref_primary, use its resolved_text
        if node.coref_primary:
            primary_resolved = node.coref_primary.resolved_text or node.coref_primary.text
            return primary_resolved
        if node.pos == Pos.PRON and node.pronoun_antecedent:
            return node.pronoun_antecedent.resolved_text or node.pronoun_antecedent.text
        return node.text
    
    def text(self) -> str:
        if not self.nodes:
            return ""
        return Vertex.resolved_text(self.nodes[0])
    
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
    
    def have_no_link(self, vertex1: Vertex, vertex2: Vertex) -> bool:
        if vertex1 == self.root:
            return True
        if vertex2 == self.root:
            return False
        node1 = self.current_node(vertex1)
        node2 = self.current_node(vertex2)
        
        subjects_dep = {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.csubjpass, Dep.agent, Dep.expl}

        if node1.dep in subjects_dep and node2.dep in subjects_dep:
            return True
        objects_dep = {Dep.dobj, Dep.iobj, Dep.pobj, Dep.dative, Dep.attr, Dep.oprd, Dep.pcomp}
        main_concept_dep =  subjects_dep | objects_dep
        
        if node1.dep in main_concept_dep and node2.dep in main_concept_dep:
            return True
        
        return False
    
    def is_sub_vertex(self, vertex1: Vertex, vertex2: Vertex) -> bool:
        # check if vertex1 is a sub-vertex of vertex2 in this hyperedge
        if vertex1 == self.root:
            return True
        if vertex2 == self.root:
            return False
        node1 = self.current_node(vertex1)
        node2 = self.current_node(vertex2)
        # b is sub-vertex of a if:
        
        subjects_dep = {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.csubjpass, Dep.agent, Dep.expl}
        objects_dep = {Dep.dobj, Dep.iobj, Dep.pobj, Dep.dative, Dep.attr, Dep.oprd, Dep.pcomp}
        main_concept_dep =  subjects_dep | objects_dep
        
        assert not (node1.dep in subjects_dep and node2.dep in subjects_dep), f"Both nodes are subjects '{node1.text}' ({node1.dep.name}) and '{node2.text}' ({node2.dep.name})"
        
        if node1.dep in subjects_dep and node2.dep not in subjects_dep:
            return True
        if node2.dep in subjects_dep and node1.dep not in subjects_dep:
            return False
        
        assert not (node1.dep in main_concept_dep and node2.dep in main_concept_dep), f"Both nodes are main '{node1.text}' ({node1.dep.name}) and '{node2.text}' ({node2.dep.name})"
        
        if node1.dep in main_concept_dep and node2.dep not in main_concept_dep:
            return True
        if node2.dep in main_concept_dep and node1.dep not in main_concept_dep:
            return False
        
        if node1.index < node2.index:
            return True
        return False

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
