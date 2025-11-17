import coreferee, spacy
from fastcoref import spacy_component
from combine import combine, calc_correfs_str
from dependency import Dependency, Node, LocalDoc
import os, glob, json

nlp0 = spacy.load('en_core_web_trf')
nlp0.add_pipe('fastcoref', 
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})

nlp1 = spacy.load('en_core_web_trf')

text = "a new software's function is to this process data. The goal of this process is the optimization."

def get_rel(nlp0, nlp1, text):
    doc1 = nlp0(text, component_cfg={"fastcoref": {'resolve_text': True}})
    print(f"Resolved Text: {doc1._.resolved_text}")
    correfs = calc_correfs_str(doc1)
    doc2 = nlp1(doc1._.resolved_text)
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
    return vertices, rel, id_map
#     print(f"Dependency Relationships:\n")
#     for r in rel:
#         print(r)

# Give all node that share the same ID a single string.


def load_ctx_texts(json_dir):
    texts = []
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                continue
            for ctx in obj.get("ctxs", []):
                t = ctx.get("text")
                if isinstance(t, str):
                    texts.append(t)
    return texts
# Example usage: set json_dir to the folder containing your JSON files
json_dir = "/home/vincent/rag/data_select/popqa/retrival/"
target_dir = "/home/vincent/rag/data_select/popqa/mygo/"
ctx_texts = load_ctx_texts(json_dir)
cnt = 0
now = 2125
for text in ctx_texts:
    if cnt < now:
        cnt += 1
        continue
    vertices, rel, id_map = get_rel(nlp0, nlp1, text)
    res = ""
    for r in rel:
        res += f"{r}\n"
    single_name: dict[int, str] = {}
    for vertex, vid in id_map.items():
        if vid not in single_name:
            single_name[vid] = vertex.text

    for vertex in vertices:
        res += f"Vertex: {vertex.text}, ID: {id_map[vertex]} ['{single_name[id_map[vertex]]}']"

    res += f"Squeezed rate is {len(single_name)}/{len(vertices)} = {len(single_name)/len(vertices):.2%}"
    with open(os.path.join(target_dir, f"{cnt}.txt"), "w", encoding="utf-8") as f:
        f.write(res)
    cnt += 1
