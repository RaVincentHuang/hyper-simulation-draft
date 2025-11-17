from hypergraph import Hypergraph, get_semantic_cluster_pairs

query_file = "query_hypergraph.pkl"
data_file = "data_hypergraph.pkl"

query_hypergraph = Hypergraph.load(query_file)
data_hypergraph = Hypergraph.load(data_file)

pairs = get_semantic_cluster_pairs(query_hypergraph, data_hypergraph)
for qc, dc, score in pairs:
    print(f"Query Cluster Text: {qc.text()}")
    print(f"Data Cluster Text: {dc.text()}")
    print(f"Similarity Score: {score:.4f}")
    print("-----")