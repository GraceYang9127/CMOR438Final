import numpy as np
import pytest
import networkx as nx

from rice_ml.unsupervised_learning.community_detection import CommunityDetector

@pytest.fixture
def karate():
    return nx.karate_club_graph()

@pytest.fixture
def two_cliques():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])        # clique A
    G.add_edges_from([(3, 4), (4, 5), (3, 5)])        # clique B
    G.add_edge(2, 3)                                   # bridge
    return G

# Greedy method
def test_greedy_fit_returns_self(karate):
    """
    Test that fit() returns the fitted CommunityDetector instance.

    Checks
    ------
    - fit() returns self.
    """
    cd = CommunityDetector(method="greedy")
    assert cd.fit(karate) is cd


def test_greedy_communities_populated(karate):
    """
    Test that communities_ is a non-empty list after fitting.

    Checks
    ------
    - communities_ is a list with at least one community.
    """
    cd = CommunityDetector(method="greedy").fit(karate)
    assert isinstance(cd.communities_, list)
    assert len(cd.communities_) >= 1


def test_greedy_labels_covers_all_nodes(karate):
    """
    Test that every node in the graph gets a community label.

    Checks
    ------
    - labels_ contains an entry for every node in G.
    """
    cd = CommunityDetector(method="greedy").fit(karate)
    assert set(cd.labels_.keys()) == set(karate.nodes())


def test_greedy_labels_are_contiguous_ints(karate):
    """
    Test that community indices are 0-indexed contiguous integers.

    Checks
    ------
    - set of label values == {0, 1, ..., n_communities - 1}.
    """
    cd = CommunityDetector(method="greedy").fit(karate)
    n = len(cd.communities_)
    assert set(cd.labels_.values()) == set(range(n))


def test_greedy_partitions_all_nodes(karate):
    """
    Test that the union of all communities equals the full node set.

    Checks
    ------
    - Union of communities_ == set of all nodes in G.
    """
    cd = CommunityDetector(method="greedy").fit(karate)
    union = set().union(*cd.communities_)
    assert union == set(karate.nodes())


def test_greedy_finds_two_groups_on_two_cliques(two_cliques):
    """
    Test that greedy method separates two obvious cliques into distinct communities.

    Checks
    ------
    - Nodes 0,1,2 get the same label; nodes 3,4,5 get the same label.
    - The two groups have different labels.
    """
    cd = CommunityDetector(method="greedy").fit(two_cliques)
    label_A = cd.labels_[0]
    label_B = cd.labels_[3]
    assert cd.labels_[1] == label_A
    assert cd.labels_[2] == label_A
    assert cd.labels_[4] == label_B
    assert cd.labels_[5] == label_B
    assert label_A != label_B

# Girvan-Newman method
def test_girvan_newman_fit_returns_self(two_cliques):
    """
    Test that fit() returns self for girvan_newman method.

    Checks
    ------
    - fit() returns self.
    """
    cd = CommunityDetector(method="girvan_newman", n_communities=2)
    assert cd.fit(two_cliques) is cd


def test_girvan_newman_n_communities(two_cliques):
    """
    Test that girvan_newman produces at least n_communities communities.

    Checks
    ------
    - len(communities_) >= n_communities.
    """
    cd = CommunityDetector(method="girvan_newman", n_communities=2).fit(two_cliques)
    assert len(cd.communities_) >= 2


def test_girvan_newman_labels_covers_all_nodes(two_cliques):
    """
    Test that every node gets a label under girvan_newman.

    Checks
    ------
    - labels_ contains an entry for every node in G.
    """
    cd = CommunityDetector(method="girvan_newman", n_communities=2).fit(two_cliques)
    assert set(cd.labels_.keys()) == set(two_cliques.nodes())



# Predict
def test_predict_known_nodes(karate):
    """
    Test that predict returns the correct community index for fitted nodes.

    Checks
    ------
    - predict([node]) matches labels_[node] for all nodes.
    """
    cd = CommunityDetector(method="greedy").fit(karate)
    nodes = list(karate.nodes())[:5]
    result = cd.predict(nodes)
    expected = np.array([cd.labels_[n] for n in nodes])
    assert np.array_equal(result, expected)


def test_predict_unknown_node_returns_minus_one(karate):
    """
    Test that predict returns -1 for a node not seen during fit.

    Checks
    ------
    - predict([unseen_node]) == [-1].
    """
    cd = CommunityDetector(method="greedy").fit(karate)
    result = cd.predict([9999])
    assert result[0] == -1


# Invalid method
def test_invalid_method_raises(karate):
    """
    Test that an unknown method raises ValueError.

    Checks
    ------
    - Passing method='bad' raises ValueError.
    """
    cd = CommunityDetector(method="bad")
    with pytest.raises(ValueError):
        cd.fit(karate)
