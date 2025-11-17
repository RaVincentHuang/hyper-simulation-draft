from sentence_transformers import SentenceTransformer
from modelscope import AutoModel, AutoTokenizer
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
# model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B')

_model_cache = {}


max_length = 8192

def get_embedding_batch_old(texts: list[str]) -> list[np.ndarray]:
    if 'Qwen3-Embedding-4B' not in _model_cache:
        _model_cache['Qwen3-Embedding-4B'] = (AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left'), AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B'))

    tokenizer, model = _model_cache['Qwen3-Embedding-4B']
    
    batch_dict = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().detach().numpy()
    embedding_list = embeddings.tolist()
    return embedding_list

def get_embedding_batch(texts: list[str], N: int=8) -> list[np.ndarray]:
    if "Qwen/Qwen3-Embedding-0.6B" not in _model_cache:
        _model_cache["Qwen/Qwen3-Embedding-0.6B"] = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model = _model_cache["Qwen/Qwen3-Embedding-0.6B"]
    embeddings = []
    for i in range(0, len(texts), N):
        batch_texts = texts[i:i+N]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.extend(batch_embeddings)
    return embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))