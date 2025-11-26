# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Rexhaif/rubert-base-srl-seqlabeling")
model = AutoModelForTokenClassification.from_pretrained("Rexhaif/rubert-base-srl-seqlabeling")

text = "Scholar Nilsson delivered a keynote at Stockholmsmässan on August."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(predictions)