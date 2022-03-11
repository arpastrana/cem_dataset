"""
This file contains a series of utility functions to convert
COMPAS CEM datastructures into Pytorch geometric ingestible data.
"""

import os
import numpy as np
from compas_cem.diagrams import TopologyDiagram


def topology_from_json(filepath):
    """
    Load a COMPAS CEM topology diagram from a JSON file.

    Parameters
    ----------
    filepath : `str`
        The file path to the location of the JSON file.

    Returns
    -------
    topology: `compas_cem.diagrams.TopologyDiagram`
        A COMPAS CEM topology diagram.
    """
    # instantiate topology diagram from json
    filepath = os.path.abspath(filepath)
    return TopologyDiagram.from_json(filepath)


def topology_to_numpy_arrays(topology, include_aux_trails=True):
    """
    Convert a COMPAS CEM topology diagram into a tuple of 3 numpy arrays.
    The arrays encode the labels of the nodes and edges of the graph.

    Parameters
    ----------
    topology: `compas_cem.diagrams.TopologyDiagram`
        A COMPAS CEM topology diagram.
    include_aux_trails: `bool`, optional
        A flag to process topology diagrams that include auxiliary trails.
        If the flat is set to `False` and the topology diagram has
        auxiliary trails, then this function will return `None` instead
        of arrays.
        Defaults to `True`.

    Returns
    -------
    X : `np.array` (n_nodes, 2)
        The node feature matrix.
        If a node is a support node, the node's feature vector is [1, 0].
        Otherwise, the vector is [0, 1].
    edge_index: `np.array` (2, n_edges)
        The adjacency matrix of the topology diagram stored in sparse format.
        An edge can only connect two nodes.
        The first row indicates the index of the first node an edge links.
        The second row encodes the index of the second node an edge connects.
    y : `np.array` (n_edges, )
        A vector with the target edge labels.
        0 indicates an edge is a trail edge.
        1 indicates if an edge is a deviation edge.
    """
    # hard-coded feature vectors
    node_type_to_vector = {"support": [0, 1], "other": [1, 0]}
    edge_type_to_label = {"trail": 0, "deviation": 1}

    # query topology stats
    n_nodes = topology.number_of_nodes()
    n_edges = topology.number_of_edges()
    n_trails_aux = topology.number_of_auxiliary_trails()

    # print stats
    print(topology.__repr__())

    # check for auxiliary trails
    if not include_aux_trails:
        print("Checking if the model has auxiliary trails...")
        if n_trails_aux > 0:
            print(f"The topology diagram has {n_trails_aux} auxiliary trails.")
            print("I am skipping it!")
            return [None] * 3
        else:
            print("This topology diagram has no auxiliary trails.")

    # create node features matrix
    # TODO: we assume all nodes have numeric keys starting at 0
    # we also assume the node keys are increasing monotonically, without gaps
    X = np.empty(shape=(n_nodes, 2), dtype=int)
    for node in topology.nodes():
        node_type = topology.node_attribute(node, "type")
        if node_type != "support":
            node_type = "other"
        X[node, :] = node_type_to_vector[node_type]

    # make edge index matrix
    edge_index = np.array(list(topology.edges()), dtype=int)

    # build edge labels array
    y = np.empty(shape=n_edges, dtype=int)
    for i, edge in enumerate(topology.edges()):
        edge_type = topology.edge_attribute(edge, "type")
        y[i] = edge_type_to_label[edge_type]

    # shallow tests
    assert edge_index.shape[0] == y.size
    assert topology.number_of_deviation_edges() == np.sum(y==edge_type_to_label["deviation"])
    assert topology.number_of_trail_edges() == np.sum(y==edge_type_to_label["trail"])
    assert topology.number_of_support_nodes() == np.sum(X[:, 0]==node_type_to_vector["support"][0])

    # return gracefully
    return X, edge_index, y


if __name__ == "__main__":

    from compas_cem.plotters import TopologyPlotter
    from compas.geometry import Rotation
    from compas.geometry import Frame

    filepath = "../bridges/zermatt/zermattB/2d/zermattB_T_2d_5_2_3_7_3_4_0.json"
    include_aux_trails = True

    # convert json file to topology to numpy arrays
    topology = topology_from_json(filepath)
    X, edge_index, y = topology_to_numpy_arrays(topology, include_aux_trails)

    # print arrays
    print(f"X:\n{X}")
    print(f"edge_index:\n{edge_index}")
    print(f"y:\n{y}")

    # visualize
    R = Rotation.from_frame_to_frame(Frame.worldZX(),
                                     Frame(point=[0.0, 0.0, 0.0],
                                           xaxis=[0.0, 1.0, 0.0],
                                           yaxis=[1.0, 0.0, 0.0]))
    topology = topology.transformed(R)

    plotter = TopologyPlotter(topology)
    plotter.draw_edges(text="key")
    plotter.draw_nodes(text="key")
    plotter.show()
