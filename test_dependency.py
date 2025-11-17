import coreferee, spacy
from fastcoref import spacy_component
from combine import combine, calc_correfs_str
from dependency import Dependency, Node, LocalDoc
# from hypergraph import Hypergraph

nlp0 = spacy.load('en_core_web_trf')
nlp0.add_pipe('fastcoref', 
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
nlp1 = spacy.load('en_core_web_trf')

# text = "Scholar Nilsson delivered a keynote at Stockholmsmässan on August. He also participated in roundtable discussions. That day, the venue hosted an AI ethics seminar, which featured his keynote and discussions."
# text = "EcoTech developed an AI application that analyzes consumption patterns to optimize power usage."
# text = "A technology company, has created a new software. This software utilizes artificial intelligence. Its function is to analyze data from users. The goal of this process is the optimization of energy consumption."
text = "Mary and John went to the market."
doc1 = nlp0(text, component_cfg={"fastcoref": {'resolve_text': True}}) # 指代消解

print(f"Resolved Text: {doc1._.resolved_text}")
print("1")
doc2 = nlp1(doc1._.resolved_text) # 消解后的再做依存分析
correfs = calc_correfs_str(doc1) # 指代消解结果
for token in doc2:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")
spans_to_merge = combine(doc2, correfs)
with doc2.retokenize() as retokenizer:
    for span in spans_to_merge:
        retokenizer.merge(span)

for token in doc2:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")

nodes, roots = Node.from_doc(doc2)
# print(f"Dependency Tree Nodes:{nodes}")
# print(f"Dependency Tree Roots:{roots}")
local_doc = LocalDoc(doc2)
dep = Dependency(nodes, roots, local_doc)
vertices, rel, id_map = dep.solve_conjunctions().mark_pronoun_antecedents().mark_prefixes().mark_vertex().compress_dependencies().calc_relationships()
print(f"Dependency Relationships:\n")
for r in rel:
    print(r)

# Give all node that share the same ID a single string.
single_name: dict[int, str] = {}
for vertex, vid in id_map.items():
    if vid not in single_name:
        single_name[vid] = vertex.text

for vertex in dep.vertexes:
    print(f"Vertex: {vertex.text}, ID: {id_map[vertex]} ['{single_name[id_map[vertex]]}']")

print(f"Squeezed rate is {len(single_name)}/{len(dep.vertexes)} = {len(single_name)/len(dep.vertexes):.2%}")

local_doc = LocalDoc(doc2)

# hypergraph = Hypergraph.from_rels(vertices, rel, id_map, local_doc)

# filename = "query_hypergraph.pkl"

# hypergraph.save(filename)
