"""
community_detection.py

Graph community detection using NetworkX.

Modularity Q measures partition quality: Q = (1/2m) * sum_{u,v} [ A_{uv} - k_u * k_v / (2m) ] * delta(c_u, c_v)
where m = number of edges, k_u = degree of node u, delta(c_u, c_v) = 1 if u and v are in the same community.

Supported methods:
  greedy: Greedy modularity maximization (Clauset-Newman-Moore).
  girvan_newman: Iteratively removes edges with highest betweenness centrality.

The reference dataset for this module is nx.karate_club_graph().

Classes
-------
CommunityDetector: Wrapper around NetworkX community detection algorithms.
"""

import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

class CommunityDetector(object):
    """
    Graph community detection using NetworkX.

    Parameters
    ----------
    method: {'greedy', 'girvan_newman'}, default='greedy'
        Algorithm to use.
        'greedy' maximizes modularity greedily.
        'girvan_newman' removes high-betweenness edges iteratively.
    n_communities : int, default=2
        Target number of communities (used only by 'girvan_newman').

    Attributes
    ----------
    communities_: list of sets
        Each set contains the node IDs in one community.
    labels_: dict
        Maps each node ID to its community index (0-indexed).
    """

    def __init__(self, method="greedy", n_communities=2):
        self.method = method
        self.n_communities = n_communities
        self.communities_ = None
        self.labels_ = None

    def fit(self, G):
        """
        Detect communities in graph G.

        Parameters
        ----------
        G: networkx.Graph
            Undirected graph to partition.

        Returns
        -------
        self

        Raises
        ------
        ImportError: If networkx is not installed.
        ValueError: If an unknown method is specified.
        """
        if not _HAS_NX:
            raise ImportError("networkx is required. Install with: pip install networkx")

        if self.method == "greedy":
            from networkx.algorithms.community import greedy_modularity_communities
            self.communities_ = list(greedy_modularity_communities(G))

        elif self.method == "girvan_newman":
            from networkx.algorithms.community import girvan_newman
            comp = girvan_newman(G)
            for communities in comp:
                self.communities_ = list(communities)
                if len(self.communities_) >= self.n_communities:
                    break
        else:
            raise ValueError(f"Unknown method: '{self.method}'. Use 'greedy' or 'girvan_newman'.")

        # Build a node -> community index mapping
        self.labels_ = {}
        for i, community in enumerate(self.communities_):
            for node in community:
                self.labels_[node] = i
        return self

    def predict(self, nodes):
        """
        Return the community index for each node in nodes.

        Parameters
        ----------
        nodes: iterable
            Node IDs to look up.

        Returns
        -------
        labels: ndarray of int
            Community index for each node. -1 if node was not seen during fit.
        """
        return np.array([self.labels_.get(n, -1) for n in nodes])
