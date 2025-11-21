from enum import Enum, IntEnum
from thefuzz import process


class Pos(IntEnum):
    ADP = 1
    ADV = 2
    ADJ = 3
    AUX = 4
    CCONJ = 5
    DET = 6
    INTJ = 7
    NOUN = 8
    NUM = 9
    PART = 10
    PRON = 11
    PROPN = 12
    PUNCT = 13
    SCONJ = 14
    SYM = 15
    VERB = 16
    X = 17
    SPACE = 18

class Tag(IntEnum):
    CC = 1
    CD = 2
    DT = 3
    EX = 4
    FW = 5
    IN = 6
    JJ = 7
    JJR = 8
    JJS = 9
    MD = 10
    NN = 11
    NNS = 12
    NNP = 13
    NNPS = 14
    POS = 15
    PRP = 16
    PRPD = 17
    RB = 18
    RBR = 19
    RBS = 20
    RP = 21
    TO = 22
    UH = 23
    VB = 24
    VBZ = 25
    VBP = 26
    VBD = 27
    VBN = 28
    VBG = 29
    WP = 30
    WPD = 31
    WRB = 32
    _SP = 33
    HYPH = 34
    ADD = 35
    WDT = 36
    PDT = 37
    XX = 38
    NFP = 39
    SYM = 40
    LS = 41
    
    WILDCARD = 99

class Dep(IntEnum):
    nsubj = 1
    nsubjpass = 2
    csubj = 3
    csubjpass = 4
    dobj = 5
    iobj = 6
    pobj = 7
    dative = 8
    
    amod = 9
    advmod = 10
    nummod = 11
    quantmod = 12
    npadvmod = 13
    neg = 14
    
    acl = 15
    advcl = 16
    ccomp = 17
    xcomp = 18
    relcl = 19
    mark = 20
    
    prep = 21
    agent = 22
    cc = 23
    conj = 24
    case = 25
    prt = 26
    
    appos = 27
    attr = 28
    acomp = 29
    oprd = 30
    aux = 31
    auxpass = 32
    expl = 33
    parataxis = 34
    meta = 35
    det = 36
    poss = 37
    predet = 38
    preconj = 39
    intj = 40
    punct = 41
    dep = 42
    
    compound = 43
    pcomp = 44
    nmod = 45

    ROOT = 46

class Entity(Enum):
    PERSON = 1
    NORP = 2
    FAC = 3
    ORG = 4
    GPE = 5
    LOC = 6
    PRODUCT = 7
    EVENT = 8
    WORK_OF_ART = 9
    LAW = 10
    LANGUAGE = 11
    DATE = 12
    TIME = 13
    PERCENT = 14
    MONEY = 15
    QUANTITY = 16
    ORDINAL = 17
    CARDINAL = 18
    NOT_ENTITY = 99

dead_dep = {Dep.dative, Dep.prt, Dep.parataxis}
solved_dep = {Dep.meta, Dep.poss, Dep.det, Dep.predet, Dep.intj}

class Node:
    def __init__(self, text: str, pos: Pos, tag: Tag, dep: Dep, ent: Entity, lemma: str, index: int) -> None:
        self.text = text
        self.original_text = text
        self.pos: Pos = pos
        self.tag: Tag = tag
        self.dep: Dep = dep
        self.ent: Entity = ent
        self.lemma: str = lemma
        self.sentence: str = text
        self.sentence_start: int = -1
        self.sentence_end: int = -1
        self.index = index
        
        self.is_vertex = False
        self.former_nodes: list[Node] = []
        
        self.dominator = False
        
        self.pronoun_antecedent: Node | None = None
        
        self.prefix_prep: str | None = None
        self.suffix_prep: str | None = None
        self.prefix_agent: str | None = None
        self.suffix_agent: str | None = None
        self.prefix_index: int | None = None
        self.suffix_index: int | None = None
        
        self.correfence_id: int | None = None
        self.is_correfence_primary: bool = False
        self.coref_primary: Node | None = None
        self.resolved_text: str | None = None

        self.head: Node | None = None
        self.children: list[Node] = []
        self.lefts: list[Node] = []
        self.rights: list[Node] = []
        
    def set_sentence(self, sentence: str, start: int, end: int) -> None:
        self.sentence = sentence
        self.sentence_start = start
        self.sentence_end = end
        
    @staticmethod
    def from_doc(doc) -> tuple[list['Node'], list['Node']]:
        nodes: list[Node] = []
        def _coref_primary_rank(node: 'Node') -> tuple[int, int, int, int]:
            ent_score = 1 if node.ent != Entity.NOT_ENTITY else 0
            pos_priority: dict[Pos, int] = {
                Pos.PROPN: 5,
                Pos.NOUN: 4,
                Pos.ADJ: 3,
                Pos.NUM: 3,
                Pos.VERB: 2,
                Pos.AUX: 2,
                Pos.PRON: 0,
            }
            pos_score = pos_priority.get(node.pos, 1)
            length_score = len(node.text)
            return (ent_score, pos_score, length_score, -node.index)
        wildcard_tags = {',', '.', '-LRB-', '-RRB-', '``', ':', "''", 'PRP$', 'WP$', '$'}
        for token in doc:
            # print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")
            pos = token.pos_
            tag = "WILDCARD" if token.tag_ in wildcard_tags else token.tag_
            dep = token.dep_
            ent = token.ent_type_ if token.ent_type_ else "NOT_ENTITY"
            sentence = doc[token.left_edge.i : token.right_edge.i + 1].text
            node = Node(
                text=token.text,
                pos=Pos[pos],
                tag=Tag[tag],
                dep=Dep[dep],
                ent=Entity[ent],
                lemma=token.lemma_,
                index=token.i,
            )
            node.set_sentence(sentence, token.left_edge.i, token.right_edge.i + 1)
            # print(f"Set sentence for Node '{node.text}' [{node.sentence_start}, {node.sentence_end}): \n\t'{node.sentence}'")
            nodes.append(node)
        
        for token, node in zip(doc, nodes):
            if token.head.i != token.i:
                node.head = nodes[token.head.i]
                nodes[token.head.i].children.append(node)
            for left in token.lefts:
                node.lefts.append(nodes[left.i])
            for right in token.rights:
                node.rights.append(nodes[right.i])
        
        # 向node里面标记指代
        if doc._.coref_clusters is not None and doc._.resolved_text is not None:
            text, resolved_text, coref_clusters = doc.text, doc._.resolved_text, doc._.coref_clusters
            # print(f"\n[Coreference Processing] Found {len(coref_clusters)} coreference cluster(s)")
            cluster_id = 0
            for cluster in coref_clusters:
                cluster_tokens: list[Node] = []
                cluster_token_set: set[Node] = set()
                cluster_texts: list[str] = []
                primary_candidates: list[Node] = []
                for start, end in cluster:
                    span_text = text[start:end]
                    cluster_texts.append(span_text)
                    # print(f"    - Span [{start}:{end}]: '{span_text}'") # 找到这轮指代中的span
                    span = doc.char_span(start, end)
                    if span is None:
                        iterable_tokens = (token for token in doc)
                    else:
                        iterable_tokens = span
                    for token in iterable_tokens:
                        token_start = token.idx
                        token_end = token.idx + len(token.text)
                        if token_end <= start or token_start >= end:
                            continue
                        node = nodes[token.i]
                        if node not in cluster_token_set:
                            cluster_tokens.append(node)
                            cluster_token_set.add(node)
                
                if len(cluster_tokens) > 1:
                    # 确定主节点：找到在resolved_text中相同位置出现的文本
                    primary_text = None
                    primary_start = None
                    primary_end = None
                    for start, end in cluster:
                        span_text = text[start:end]
                        if span_text in resolved_text:
                            resolved_pos = resolved_text.find(span_text)
                            if resolved_pos != -1:
                                primary_text = span_text
                                primary_start = start
                                primary_end = end
                                break
                    # if primary_text:
                        # print(f"    Primary text: '{primary_text}' at [{primary_start}:{primary_end}]")
                    # else:
                        # print(f"    Warning: No primary text found for cluster {cluster_id}")
                    primary_node: Node | None = None
                    for node in cluster_tokens:
                        node.correfence_id = cluster_id
                        # Check if this node is in the primary span
                        token = doc[node.index]
                        token_start = token.idx
                        token_end = token.idx + len(token.text)
                        is_primary = not (token_end <= primary_start or token_start >= primary_end) if primary_start is not None and primary_end is not None else False
                        if is_primary:
                            node.is_correfence_primary = True
                            primary_candidates.append(node)
                        # print(f"      - Node '{node.text}' (index={node.index}): correfence_id={cluster_id}, is_primary={is_primary}")
                    
                    primary_text_for_replacement: str | None = None
                    if primary_text:
                        primary_text_for_replacement = primary_text
                    elif cluster_texts:
                        primary_text_for_replacement = cluster_texts[0]

                    if primary_candidates:
                        primary_node = max(primary_candidates, key=_coref_primary_rank)
                        for candidate in primary_candidates:
                            candidate.is_correfence_primary = candidate is primary_node
                    if not primary_node and cluster_tokens:
                        primary_node = max(cluster_tokens, key=_coref_primary_rank)
                    if not primary_text_for_replacement and primary_node:
                        primary_text_for_replacement = primary_node.text
                    
                    for node in cluster_tokens:
                        if node.is_correfence_primary:
                            node.resolved_text = node.text
                            continue
                        if node.pos == Pos.PRON and primary_text_for_replacement:
                            node.resolved_text = primary_text_for_replacement
                        if primary_node:
                            node.coref_primary = primary_node
                            if node.pos == Pos.PRON and not node.pronoun_antecedent:
                                node.pronoun_antecedent = primary_node
                        elif node.pos == Pos.PRON and not node.pronoun_antecedent:
                            node.pronoun_antecedent = None
                    if primary_node:
                        if not primary_text_for_replacement:
                            primary_text_for_replacement = primary_node.text
                        if not primary_node.resolved_text:
                            primary_node.resolved_text = primary_node.text
                        if not primary_node.is_correfence_primary:
                            primary_node.is_correfence_primary = True
                    
                    cluster_id += 1
                else:
                    print(f"    Skipping cluster {cluster_id} (only {len(cluster_tokens)} token node, need > 1)")
                    cluster_id += 1
            print(f"\n[Coreference Processing] Completed. Total clusters processed: {cluster_id}\n")
        
        roots = [node for node in nodes if node.head is None]
        for root in roots:
            assert root.dep == Dep.ROOT or root.dep == Dep.dep, f"Root node dep should be ROOT or _SP, got {root.dep.name}"
        return nodes, roots
    
    def __format__(self, format_spec: str) -> str:
        return f"Node(text='{self.text}', pos={self.pos.name}, tag={self.tag.name}, dep={self.dep.name}, ent={self.ent.name}, sentence='{self.sentence}')"
    def __repr__(self) -> str:
        return self.__format__('')
    def __display__(self) -> str:
        return self.__format__('')
    def __str__(self) -> str:
        return self.__format__('')

class Relationship:
    def __init__(self, entities: list[Node], sentence: str, relationship_sentence: str) -> None:
        self.root = entities[0]
        self.entities = entities
        self.sentence = sentence
        self.relationship_sentence = relationship_sentence
        # start, end are the range of entities' indices
        start, end = entities[0].index, entities[0].index
        for entity in entities[1:]:
            if entity.index < start:
                start = entity.index
            if entity.index > end:
                end = entity.index
        self.start = start
        self.end = end + 1
        self.father: Relationship | None = None

    @staticmethod
    def _resolved_node_text(node: Node) -> str:
        if node.resolved_text:
            return node.resolved_text
        if node.pos == Pos.PRON and node.coref_primary:
            return node.coref_primary.text
        if node.pos == Pos.PRON and node.pronoun_antecedent:
            return node.pronoun_antecedent.text
        return node.text

    def node_text(self, node: Node) -> str:
        res = self._resolved_node_text(node)

        determiner_children: list[Node] = []
        for child in node.lefts:
            if child.dep in {Dep.det, Dep.poss, Dep.predet}:
                determiner_children.append(child)
        determiner_children.sort(key=lambda n: n.index)
        if determiner_children:
            prefix = " ".join(self._resolved_node_text(child) for child in determiner_children)
            res = f"{prefix} {res}"

        return res
    
    def relationship_text_simple(self) -> str:
        
        # `self.relation_sentence` is a part of `self.sentence`
        # We need firstly record the prefix and suffix of `self.relation_sentence` to `self.sentence` 
        # after `sentence` calculation, we need cancel the prefix and suffix in `sentence`
        
        # 1. calc the prefix and suffix
        def calc_prefix_suffix():
            rel_start = self.sentence.find(self.relationship_sentence)
            if rel_start != -1:
                prefix = self.sentence[:rel_start].strip()
                suffix = self.sentence[rel_start + len(self.relationship_sentence):].strip()
            else:
                prefix = ""
                suffix = ""
            return prefix, suffix
        
        prefix, suffix = calc_prefix_suffix()
        
        sentence = str(self.sentence)
        for entity in self.entities[1:]:
            new_text = self.node_text(entity)
            old_candidates = [entity.sentence, self._resolved_node_text(entity)]
            for old in old_candidates:
                if old and old in sentence:
                    sentence = sentence.replace(old, new_text, 1)
                    break
        
        sentence = sentence.replace(prefix, "").replace(suffix, "").strip()
        
        return sentence
    
    @staticmethod
    def node_text_for_fuzzy(node: Node) -> str:
        if node.pos == Pos.PRON and node.pronoun_antecedent:
            return node.pronoun_antecedent.text.lower()
        return node.text.lower()

    def __format__(self, format_spec: str) -> str:
        return f"[root: {self.node_text(self.root)}] ({', '.join([self.node_text(entity) for entity in self.entities])})\n\tIn Sentence: '{self.sentence}'\n\tSimple: '{self.relationship_text_simple()}'"
    def __repr__(self) -> str:
        return self.__format__('')
    def __str__(self) -> str:
        return self.__format__('')
    def __display__(self) -> str:
        return self.__format__('')

class LocalDoc:
    def __init__(self, doc) -> None:
        self.tokens = [token.text for token in doc]

    def __getitem__(self, index) -> str:
        if isinstance(index, slice):
            return ' '.join(self.tokens[index])
        else:
            return self.tokens[index]

class Dependency:
    def __init__(self, nodes: list[Node], roots: list[Node], doc: LocalDoc) -> None:
        self.nodes = nodes
        self.roots = roots
        self.doc = doc
        self.vertexes: list[Node] = []
        self.links_succ: dict[Node, list[Node]] = {}
        self.links_pred: dict[Node, Node] = {}
        self.relationship_sentences: dict[Node, str] = {}
        self.correfence_map: dict[Node, Node] = {}
    
    def _fixup_lefts_rights_sentences(self, node: Node) -> None:
        node.children.sort(key=lambda n: n.index)
        node.lefts = [child for child in node.children if child.index < node.index]
        node.rights = [child for child in node.children if child.index > node.index]
        node.sentence_start = node.lefts[0].index if node.lefts else node.index
        node.sentence_end = node.rights[-1].index + 1 if node.rights else node.index + 1
        # print(f"Sentence of Node '{node.text}' [{node.sentence_start}, {node.sentence_end}): \n\t'{node.sentence}'")
        node.sentence = self.doc[node.sentence_start : node.sentence_end]
        # print(f"Reset sentence of [{node.sentence_start}, {node.sentence_end}): \n\t'{node.sentence}'")
    
    def _calc_relationship_sentence(self, root: Node):
        left_edge = right_edge = root.index
        for succ in self.links_succ.get(root, []):
            if succ.index < left_edge:
                left_edge = succ.index
            if succ.index > right_edge:
                right_edge = succ.index
        return self.doc[left_edge : right_edge + 1]
    
    # PASS 1: Solve all the Conjunction dependencies
    # Solve the `conj`, `cc`, `appos` and `preconj` dependencies by change the conjunct children to the children of the head;
    # e.g., in "Alice and Bob went to the store", "Bob" will get the same children as "Alice"
    # Execute this pass by Level-Order Traversal
    def solve_conjunctions(self):
        print("Solving conjunctions...\n")
        queue = self.roots.copy()
        next_level: list[Node] = []
        while queue:
            node = queue.pop(0)
            remove_children: list[Node] = []
            for child in node.children:
                if child.dep == Dep.conj or (child.dep == Dep.appos and node.head):
                    child.dep = node.dep
                    child.head = node.head
                    remove_children.append(child)
                    queue.append(child)
                else:
                    next_level.append(child)
            
            for child in remove_children:
                node.children.remove(child)
                if node.head:
                    node.head.children.append(child)
                
            
            if not queue:
                queue = next_level
                next_level = []
            
            if not remove_children:
                continue
            
            self._fixup_lefts_rights_sentences(node)
            if node.head:
                self._fixup_lefts_rights_sentences(node.head)
        
        self.roots = [node for node in self.nodes if node.head is None]
        # print("Conjunctions solved. Resulting Nodes:")
        # for node in self.nodes:
        #     print(f"{node}")
        return self
    
    # PASS 2: Mark all the antecedent of pronouns.
    # check all the `relcl` dependencies, uses the relative clause to find the antecedent of pronouns.
    # We split all `relcl` conditions to two dependencies tree.
    def mark_pronoun_antecedents(self):
        for node in self.nodes:
            if node.dep != Dep.relcl or not node.head:
                continue
            antecedent = node.head
            for child in node.children:
                if child.pos == Pos.PRON:
                    child.pronoun_antecedent = antecedent
            
            node.head.children.remove(node)
            self._fixup_lefts_rights_sentences(node.head)

            node.dep = Dep.ROOT
            node.head = None
        return self
    
    # PASS 3: Mark the prefixes for prepositions and agents
    def mark_prefixes(self):
        for node in self.nodes:
            if node.dep == Dep.agent and node.head:
                if node.index < node.head.index:
                    node.head.prefix_agent = node.text
                    node.head.prefix_index = node.index
                else:
                    node.head.suffix_agent = node.text
                    node.head.suffix_index = node.index
            if node.dep == Dep.prep and node.head:
                if node.index < node.head.index:
                    node.head.prefix_prep = node.text
                    node.head.prefix_index = node.index
                else:
                    node.head.suffix_prep = node.text
                    node.head.suffix_index = node.index
            if node.dep == Dep.pobj and node.head:
                if node.index > node.head.index:
                    node.prefix_prep = node.head.text
                    node.prefix_index = node.head.index
                else:
                    node.suffix_prep = node.head.text
                    node.suffix_index = node.head.index
        return self
    
    # PASS 4: Mark all the vertex, that all the nodes should be a Vertex i.f.f. statisfy:
    # 0. All root nodes
    # 1. holds a entity, i.e., ent != NOT_ENTITY
    # 2. to be a noun or proper noun, i.e., pos in {NOUN, PROPN}
    # 3. to be a verb or auxiliary verb, i.e., pos in {VERB, AUX}
    # 4. to be an adjective, i.e., pos == ADJ
    # 5. to be a numeric, i.e., pos == NUM
    # 6. to be a pronoun, i.e., pos == PRON
    # Merge coreference nodes: nodes with same correfence_id share the same vertex (primary node)
    def mark_vertex(self): #
        for node in self.nodes:
            if node.dep in {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.csubjpass}:
                node.dominator = True

        correfence_primary_map: dict[int, Node] = {}
        for node in self.nodes:
            if node.correfence_id is not None and node.is_correfence_primary:
                correfence_primary_map[node.correfence_id] = node

        self.correfence_map = {
            node: correfence_primary_map[node.correfence_id]
            for node in self.nodes
            if node.correfence_id is not None
            and not node.is_correfence_primary
            and node.correfence_id in correfence_primary_map
        }

        qualifying_pos = {Pos.NOUN, Pos.PROPN, Pos.VERB, Pos.AUX, Pos.ADJ, Pos.NUM, Pos.PRON}
        self.vertexes = []
        for node in self.nodes:
            node.is_vertex = False
            if (
                node.head is None
                or node.ent != Entity.NOT_ENTITY
                or node.pos in qualifying_pos
            ):
                node.is_vertex = True
                self.vertexes.append(node)
        
        return self
    
    # PASS 5: Compress dependencies that only links vertex nodes.
    # We use links to record the compressed dependencies.
    # collect all the pred non-vertex nodes of a node between vertexes into `former_nodes`.
    def compress_dependencies(self):
        for node in self.nodes:
            if node.head and node.head in self.correfence_map:
                node.head = self.correfence_map[node.head]
        for node in self.vertexes:
            if not node.head:
                continue
            pred = node.head
            while pred and not pred.is_vertex:
                if pred in self.correfence_map:
                    pred = self.correfence_map[pred]
                else:
                    node.former_nodes.insert(0, pred)  # 插入到头部
                    pred = pred.head
            if pred:
                self.links_pred[node] = pred
                if pred not in self.links_succ:
                    self.links_succ[pred] = []
                self.links_succ[pred].append(node)
        return self

    # PASS 6: Calculate all the relationships.
    # For each non-leaf vertex, it and its successors form a relationship.
    # We temporarily using the roots's sentence as the relationship sentence.
    # Then we use the `thefuzz` to get all vertex a id, and calculate a map.
    def calc_relationships(self) -> tuple[list[Node], list[Relationship], dict[Node, int]]:
        def _match_same(best_match, score, node: Node, choices_map: dict[str, int], pos_map: dict[int, Pos]) -> bool:
            if score >= 90 and (pos_map[choices_map[best_match]] == node.pos):
                return True
            elif score == 100:
                return True
            return False
        
        # for node in self.vertexes:
        #     print(f"Vertex Node: {node}")
        
        saved_rels: set[tuple[str, str]] = set()
        relationships: list[Relationship] = []
        vertex_id_map: dict[Node, int] = {}
        root_to_relationship: dict[Node, Relationship] = {}
        for node in self.vertexes:
            if node in self.links_succ:
                node_key_text = (node.resolved_text or node.text)
                if (node_key_text, node.sentence) in saved_rels:
                    continue
                relational_sentence = self._calc_relationship_sentence(node)
                saved_rels.add((node_key_text, node.sentence))
                rel = Relationship(entities=[node] + self.links_succ[node], sentence=node.sentence, relationship_sentence=relational_sentence)
                root_to_relationship[node] = rel
                relationships.append(rel)
        
        relationship_trees: dict[Relationship, Relationship] = {} # mapping relationship to its father relationship
        for rel in relationships:
            node = rel.root
            for succ in self.links_succ.get(node, []):
                if succ in root_to_relationship:
                    relationship_trees[root_to_relationship[succ]] = rel
        
        for rel, father_rel in relationship_trees.items():
            rel.father = father_rel
        
        choices = []
        choices_map: dict[str, int] = {}
        pos_map: dict[int, Pos] = {}
        cnt = 1
        deferred_coref_nodes: list[Node] = []
        for node in self.vertexes:
            if node.coref_primary:
                deferred_coref_nodes.append(node)
                continue
            base_text = node.resolved_text or node.text
            text = base_text.lower()
            extraction = process.extractOne(text, choices) if choices else None
            match extraction:
                case (best_match, score) if _match_same(best_match, score, node, choices_map, pos_map):
                    vertex_id_map[node] = choices_map[best_match]
                    pos_map[vertex_id_map[node]] = node.pos
                case _:
                    choices.append(text)
                    choices_map[text] = cnt
                    vertex_id_map[node] = cnt
                    pos_map[cnt] = node.pos
                    cnt += 1
        for node in deferred_coref_nodes:
            primary = node.coref_primary
            if primary and primary in vertex_id_map:
                vertex_id_map[node] = vertex_id_map[primary]
                pos_map[vertex_id_map[node]] = primary.pos
                continue
            base_text = node.resolved_text or node.text
            text = base_text.lower()
            extraction = process.extractOne(text, choices) if choices else None
            match extraction:
                case (best_match, score) if _match_same(best_match, score, node, choices_map, pos_map):
                    vertex_id_map[node] = choices_map[best_match]
                    pos_map[vertex_id_map[node]] = node.pos
                case _:
                    choices.append(text)
                    choices_map[text] = cnt
                    vertex_id_map[node] = cnt
                    pos_map[cnt] = node.pos
                    cnt += 1

        return self.vertexes, relationships, vertex_id_map

