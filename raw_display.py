import spacy
from spacy import displacy
import spacy



nlp1 = spacy.load('en_core_web_trf')

text = "A new software's function is to this process data."
doc = nlp1(text)
for token in doc:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")
print("\nMerging spans...\n")


displacy.serve(doc, style="dep")
