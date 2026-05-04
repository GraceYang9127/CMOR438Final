# Community Detection

Community detection finds groups of nodes in a graph that are more densely connected to each other than to the rest of the network. Two algorithms are implemented: greedy modularity maximization and Girvan-Newman.

---

## Algorithms

**Modularity** measures partition quality:

$$Q = \frac{1}{2m} \sum_{u,v} \left[ A_{uv} - \frac{k_u k_v}{2m} \right] \delta(c_u, c_v)$$

$Q > 0.3$ indicates meaningful community structure.

**Greedy (Clauset-Newman-Moore):** starts with each node in its own community, merges pairs that give the largest gain in $Q$. Fast: $O(m \log n)$.

**Girvan-Newman:** iteratively removes the edge with the highest betweenness centrality. Slower but reveals hierarchical structure.

---

## Dataset

**Food Similarity Graph** (built from `FOOD-DATA-GROUP1.csv`) - 551 food items connected by edges when their cosine similarity exceeds 0.70. At this threshold the graph has ~7700 edges and 22 connected components. Communities correspond to nutritionally similar food groups (e.g. meats, dairy, vegetables).

Falls back to the **Karate Club Graph** (`nx.karate_club_graph()`) if CSV is not found: 34 nodes, 78 edges, ground truth available.

---

## Notebook Covers

- Two-community bridge concept diagram
- Degree distribution histogram and degree vs betweenness scatter (EDA)
- Raw graph visualization
- Community detection with greedy modularity
- Modularity score and community size summary
- Colored community visualization
- Food community samples (or karate club ground truth comparison as fallback)
