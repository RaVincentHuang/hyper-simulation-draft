import numpy as np

# Try to import faiss; if not available, keep as None and provide a helpful message
try:
	import faiss
except Exception:
	faiss = None


def build_faiss_index(embs: np.ndarray, ids):
	"""Build a FAISS IndexIDMap using (embs, ids).

	Args:
		embs: np.ndarray, shape (n, d), dtype float32 (or convertible)
		ids: iterable of int (length n)

	Returns:
		faiss index
	"""
	if faiss is None:
		raise ImportError("faiss is not installed. Install with: pip install faiss-cpu")

	embs = np.asarray(embs, dtype="float32")
	# L2-normalize so IndexFlatIP behaves like cosine similarity
	norms = np.linalg.norm(embs, axis=1, keepdims=True)
	embs = embs / np.clip(norms, 1e-12, None)

	d = embs.shape[1]
	embs = np.ascontiguousarray(embs)
	ids_arr = np.asarray(list(ids), dtype="int64")
	orig = faiss.IndexFlatIP(d)
	id_index = faiss.IndexIDMap(orig)
	id_index.add_with_ids(embs, ids_arr, n)
	return id_index


def search_faiss(index, query_vec: np.ndarray, k: int = 5):
	"""Search FAISS index. Returns (ids, scores).

	Scores are inner products; for normalized vectors they approximate cosine similarity.
	"""
	q = np.asarray(query_vec, dtype="float32")
	q = q / max(1e-12, np.linalg.norm(q))
	D, I = index.search(q.reshape(1, -1), k)
	return I[0].tolist(), D[0].tolist()


if __name__ == "__main__":
	# demo: build an index and query it. Falls back to a message if faiss not installed.
	n, d = 1000, 128
	embs = np.random.randn(n, d).astype("float32")
	ids = list(range(10000, 10000 + n))

	if faiss is None:
		print("faiss not installed. To use FAISS install: pip install faiss-cpu")
		print("Demo: returning nearest by simple numpy cosine for the first 5 items")
		q = np.random.randn(d).astype("float32")
		q = q / np.linalg.norm(q)
		embs_norm = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12, None)
		sims = embs_norm @ q
		idxs = np.argsort(-sims)[:5]
		for i in idxs:
			print(f"id={ids[i]}  cosine={float(sims[i]):.4f}")
	else:
		idx = build_faiss_index(embs, ids)
		q = np.random.randn(d).astype("float32")
		ids_out, scores = search_faiss(idx, q, k=5)
		print("ids:", ids_out)
		print("scores:", scores)
