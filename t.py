import coreferee, spacy
from fastcoref import spacy_component

t3 = "Scholar Nilsson delivered a keynote at Stockholmsmässan on August. He also participated in roundtable discussions. That day, the venue hosted an AI ethics seminar, which featured the keynote and discussions."
t4 = "Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much."
t0 = "My cat."

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('fastcoref', 
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
doc = nlp(t3, component_cfg={"fastcoref": {'resolve_text': True}})
# doc._.coref_chains.print()
print(doc._.coref_clusters)
print(doc._.resolved_text)
