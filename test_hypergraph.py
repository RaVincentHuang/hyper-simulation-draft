from hypergraph import Hypergraph
from sc import get_semantic_cluster_pairs, SemanticCluster, get_d_match


query_file = "query_hypergraph.pkl"
data_file = "data_hypergraph.pkl"

query_hypergraph = Hypergraph.load(query_file)
data_hypergraph = Hypergraph.load(data_file)

pairs = get_semantic_cluster_pairs(query_hypergraph, data_hypergraph)
for qc, dc, score in pairs:
    print(f"Query Cluster Text: {qc.text()}, qc len: {sum(len(he.vertices) for he in qc.hyperedges)}")
    print(f"Vertices in Query Cluster:[{', '.join(v.text() for v in qc.get_vertices())}]")
    print(f"Data Cluster Text: {dc.text()}, dc len: {sum(len(he.vertices) for he in dc.hyperedges)}")
    print(f"Vertices in Data Cluster:[{', '.join(v.text() for v in dc.get_vertices())}]")
    print(f"Similarity Score: {score:.4f}")
    _ = get_d_match(qc, dc)
    print("-----")