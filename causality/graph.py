import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np


def draw_network(nodes, edges, cpds, observed):
    """
    To view the results of the drawing, run this script separately: peepo/visualize/server.py
    and go to: http://localhost:8000/index.html

    :param nodes:
    :param edges:
    :param cpds:
    :param observed:
    :return:
    """
    G = nx.DiGraph()

    for node in nodes:
        cpd = ''
        if node in observed:
            cpd = '1'
        elif node in cpds:
            cpd = str(np.amax(cpds[node].values))
        G.add_node(node, name=node, cpd=cpd)

    G.add_edges_from(edges)

    d = json_graph.node_link_data(G)  # node-link format to serialize

    json.dump(d, open('static/network.json', 'w'))
