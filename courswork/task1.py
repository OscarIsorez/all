import lzma
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
from Bio.Phylo.TreeConstruction import DistanceCalculator
import pandas as pd
import igraph
import itertools

# Load protein sequences
file_path = "data/sequences.txt.xz"
sequences = []

try:
    with lzma.open(file_path, "rt") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                seq_record = SeqRecord(Seq(line), id=f"seq{i+1}")
                sequences.append(seq_record)
except Exception as e:
    print(f"Error reading file: {e}")

if not sequences:
    print("No sequences loaded. Please check the file path and content.")
    exit()
else:
    print(f"Loaded {len(sequences)} sequences.")

# Compute pairwise distance matrix
calculator = DistanceCalculator('identity')
alignment_tool = Align.PairwiseAligner()
alignment_tool.mode = 'global'

# Create the pairwise distance matrix with float data type
num_sequences = len(sequences)
distance_matrix = pd.DataFrame(0.0, index=range(num_sequences), columns=range(num_sequences))

for i, j in itertools.combinations(range(num_sequences), 2):
    seq1 = str(sequences[i].seq)
    seq2 = str(sequences[j].seq)
    score = alignment_tool.score(seq1, seq2)
    max_length = max(len(seq1), len(seq2))
    identity = score / max_length
    distance = 1 - identity
    distance_matrix.iloc[i, j] = distance
    distance_matrix.iloc[j, i] = distance

# Threshold-based graph construction
threshold = 0.1  # Set a similarity cutoff
edges = [(i, j) for i, j in itertools.combinations(range(num_sequences), 2)
         if distance_matrix.iloc[i, j] <= threshold]

# Build the network graph
graph = igraph.Graph(edges=edges, directed=False)
graph.vs["label"] = [seq.id for seq in sequences]

# Analyze network properties
network_stats = {
    "Diameter": graph.diameter(),
    "Girth": graph.girth(),
    "Radius": graph.radius(),
    "Average Path Length": graph.average_path_length(),
    "Density": graph.density(),
    "Clustering Coefficient": graph.transitivity_avglocal_undirected(),
    "Connected Components": len(graph.components())
}
print("Network Statistics:")
print(pd.DataFrame([network_stats]))

# Perform clustering
clusters = graph.community_multilevel()
graph.vs["membership"] = clusters.membership

# Visualize the graph with clusters
import seaborn as sns

cluster_colors = sns.color_palette("husl", len(clusters))
graph.vs["color"] = [cluster_colors[m] for m in clusters.membership]

layout = graph.layout("fr")
igraph.plot(graph, "protein_network.png", layout=layout, bbox=(1280, 960))

# Save cluster information
cluster_df = pd.DataFrame({
    "Sequence ID": [seq.id for seq in sequences],
    "Cluster": clusters.membership
})
cluster_df.to_csv("protein_clusters.csv", index=False)

print("Clustering completed. Network visualization saved as 'protein_network.png'.")
print("Cluster information saved as 'protein_clusters.csv'.")
