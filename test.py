from math import pi
import spacy
from spacy import displacy
import coreferee, spacy
from fastcoref import spacy_component
t3 = "Scholar Nilsson delivered a keynote at Stockholmsmässan on August. He also participated in roundtable discussions. That day, the venue hosted an AI ethics seminar, which featured his keynote and discussions."

t0 = "My cat."

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--text', type=str, default=t3, help='Input text to process')
# args = parser.parse_args()
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('fastcoref', 
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
# doc = nlp(t3, component_cfg={"fastcoref": {'resolve_text': True}})


while True:
    text = input("Enter text (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    doc1 = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
    print(f"Resolved Text: {doc1._.resolved_text}")
    doc2 = nlp(doc1._.resolved_text)

    for token in doc2:
        print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")

    for chunk in doc2.noun_chunks:
        print(f"chunk.text: {chunk.text}")
    for ent in doc2.ents:
        print(f"ent.text: {ent.text}, ent.label_: {ent.label_}")


# displacy.serve(doc2, style="dep")
# # Tokens
# print("Tokens:")
# for token in doc:
#     print(f"Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}, TAG: {token.tag_}, Dep: {token.dep_}, Ent: {token.ent_type_}, Shape: {token.shape_}, Morph: {token.morph}, [ALPHA, STOP]: [{token.is_alpha}, {token.is_stop}]")

# # Noun Chunks
# print("\nNoun Chunks:")
# for chunk in doc.noun_chunks:
#     print(f"chunk.text: {chunk.text}, chunk.root.text: {chunk.root.text}, chunk.root.dep_: {chunk.root.dep_}, chunk.root.head.text: {chunk.root.head.text}")

# # Navigating the parse tree
# print("\nDependency Parse:")
# for token in doc:
#     print(f"Token: {token.text}, Dep: {token.dep_}, Head: {token.head.text}, Head POS: {token.head.pos_}, Children: {[child.text for child in token.children]}")

# # Left and right
# print("\nLeft and Right Children:")
# for token in doc:
#     print(f"Token: {token.text}, Lefts: {[left.text for left in token.lefts]}, Rights: {[right.text for right in token.rights]}")

# roots = [token for token in doc if token.head == token]
# print(f"\nRoot of the sentence: {[root.text for root in roots]}")
# for root in roots:
#     for descendant in root.subtree:
#         print(f"Descendant of root '{root.text}': {descendant.text}")
#         for ancestor in descendant.ancestors:
#             print(f"\tAncestor of '{descendant.text}': {ancestor.text}")

# for root in roots:
#     span = doc[root.left_edge.i : root.right_edge.i + 1]
    

# Display the dependency parse
# displacy.serve(doc, style="dep")
