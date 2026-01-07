from platform import node
from dependency import LocalDoc, Node, Pos, Entity, Relationship, Dep, Tag
import itertools

import index

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
    
    def tag_range(self, tag: 'Tag') -> bool:
        """判断 self 的 tag 是否与给定 tag 在同一范围内（细粒度）
        
        Tag 范围分组：
        1. 名词类: NN, NNS, NNP, NNPS
        2. 动词类: VB, VBZ, VBP, VBD, VBN, VBG
        3. 形容词类: JJ, JJR, JJS
        4. 副词类: RB, RBR, RBS
        5. 代词类: PRP, PRPD, WP, WPD
        6. 限定词类: DT, PDT, WDT
        7. 数词类: CD
        8. Wh-词类: WP, WPD, WRB, WDT
        """
        from dependency import Tag
        
        noun_tags = {Tag.NN, Tag.NNS, Tag.NNP, Tag.NNPS}
        verb_tags = {Tag.VB, Tag.VBZ, Tag.VBP, Tag.VBD, Tag.VBN, Tag.VBG}
        adj_tags = {Tag.JJ, Tag.JJR, Tag.JJS}
        adv_tags = {Tag.RB, Tag.RBR, Tag.RBS}
        pron_tags = {Tag.PRP, Tag.PRPD, Tag.WP, Tag.WPD}
        det_tags = {Tag.DT, Tag.PDT, Tag.WDT}
        num_tags = {Tag.CD}
        
        self_tags = {n.tag for n in self.nodes}
        
        if tag in noun_tags:
            return bool(self_tags & noun_tags)
        elif tag in verb_tags:
            return bool(self_tags & verb_tags)
        elif tag in adj_tags:
            return bool(self_tags & adj_tags)
        elif tag in adv_tags:
            return bool(self_tags & adv_tags)
        elif tag in pron_tags:
            return bool(self_tags & pron_tags)
        elif tag in det_tags:
            return bool(self_tags & det_tags)
        elif tag in num_tags:
            return bool(self_tags & num_tags)
        else:
            return tag in self_tags
    
    def is_domain(self, other: 'Vertex') -> bool:
        self_has_ent = any(e != Entity.NOT_ENTITY for e in self.ents)
        other_has_ent = any(e != Entity.NOT_ENTITY for e in other.ents)
        if self_has_ent and other_has_ent:
            return any(self.ent_range(e) for e in other.ents if e != Entity.NOT_ENTITY)
        if self_has_ent != other_has_ent:
            return False
        return any(self.pos_range(p) for p in other.poses) # more strict：pose_same
        # 再加一层word_net
        
    def is_domain_pos2tag(self, other: 'Vertex') -> bool:
        """Same logic as is_domain, but replace pos_range with tag_range in fallback."""
        self_has_ent = any(e != Entity.NOT_ENTITY for e in self.ents)
        other_has_ent = any(e != Entity.NOT_ENTITY for e in other.ents)
        if self_has_ent and other_has_ent:
            return any(self.ent_range(e) for e in other.ents if e != Entity.NOT_ENTITY)
        if self_has_ent != other_has_ent:
            return False
        # Use tag_range instead of pos_range
        return any(self.tag_range(n.tag) for n in other.nodes)

    def is_domain_tag(self, other: 'Vertex') -> bool:
        # TAG range
        if any(self.tag_range(n.tag) for n in other.nodes):
            return True
        # POS range
        # if any(self.pos_range(p) for p in other.poses):
            # return True
        # ENT range
        if any(self.ent_range(e) for e in other.ents):
            return True
        return False

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
