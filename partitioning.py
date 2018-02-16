
import argparse
import logging
from collections import Counter

import numpy as np


def import_nodes_from_tg(tg_file):
    edges = []
    nodes = set()
    nodes_map = dict()

    with open(tg_file) as f:
        lines = f.readlines()
        nodes_size = int(lines[0]) + 2
        adjacency_matrix = np.zeros((nodes_size, nodes_size))
        for line in lines:
            node = line.split()
            node_id = int(node[0])
            nodes.add(node_id)
            nodes_map[node_id] = node_id
            for i in range(4, len(node)):
                edges.append((node_id, int(node[i])))
                adjacency_matrix[int(node[0])][int(node[i])] = 1
                adjacency_matrix[int(node[i])][int(node[0])] = 1

    logging.info("Imported {} nodes with {} edges from {}".format(nodes_size, len(edges), tg_file))
    return nodes, edges, adjacency_matrix, nodes_map


def import_nodes(nodes_file):
    """
    Import the nodes from the file
    """

    edges = []
    nodes = set()
    nodes_map = dict()
    latest_node = 0

    with open(nodes_file) as f:
        lines = f.readlines()
        for line in lines:
            v1, v2 = line.split()
            if v1 not in nodes_map:
                nodes_map[v1] = latest_node
                nodes.add(latest_node)
                latest_node += 1

            if v2 not in nodes_map:
                nodes_map[v2] = latest_node
                nodes.add(latest_node)
                latest_node += 1

            edges.append((nodes_map[v1], nodes_map[v2]))

    number_nodes = len(nodes)
    logging.info("Imported {} nodes with {} edges from {}".format(number_nodes, len(edges), nodes_file))

    adjacency_matrix = np.zeros((number_nodes, number_nodes))
    for v1, v2 in edges:
        adjacency_matrix[v1][v2] = 1
        adjacency_matrix[v2][v1] = 1

    return nodes, edges, adjacency_matrix, nodes_map


def degree_nodes(adjacency_matrix, number_nodes):
    """
    Compute the degree of each node
    Returns the vector of degrees
    """

    d = []
    for i in range(number_nodes):
        d.append(sum([adjacency_matrix[i][j] for j in range(number_nodes)]))

    return d


def get_min_cuts(edges, edges_in_between):
    logging.info('total edges {} in between {}'.format(len(edges), len(edges_in_between)))
    nodes = []
    for v1, v2 in edges_in_between:
        nodes.append(v1)
        nodes.append(v2)
    new_edges = []
    node = Counter(nodes).most_common()[0][0]
    for v1, v2 in edges:
        if v1 != node and v2 != node:
            new_edges.append((v1, v2))
    new_edges_in_between = []
    for edge in edges_in_between:
        if edge in new_edges:
            new_edges_in_between.append(edge)
    if len(new_edges_in_between) > 0:
        return get_min_cuts(new_edges, new_edges_in_between)
    else:
        return new_edges


def get_nodes(edges):
    nodes = set()
    for v1, v2 in edges:
        nodes.add(v1)
        nodes.add(v2)
    return nodes


def partition_graph(nodes_file, tg=False):
    logging.debug("Computing Adjacency Matrix...")

    nodes, edges, adjacency_matrix, nodes_map = import_nodes_from_tg(nodes_file) if tg else import_nodes(nodes_file)
    logging.debug("Adjacency matrix:\n", adjacency_matrix)

    logging.debug("Computing the degree of each node...")
    degrees = degree_nodes(adjacency_matrix, len(nodes))
    logging.debug("Degrees: ", degrees)

    logging.debug("Computing the Laplacian matrix...")
    laplacian_matrix = np.diag(degrees) - adjacency_matrix
    logging.debug("Laplacian matrix:\n", laplacian_matrix)

    logging.debug("Computing the eigenvectors and eigenvalues...")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    logging.debug("Found eigenvalues: ", eigenvalues)

    # Index of the second eigenvalue
    index_fnzev = np.argsort(eigenvalues)[1]

    logging.debug("Eigenvector for #{} eigenvalue ({}): ".format(
        index_fnzev, eigenvalues[index_fnzev]), eigenvectors[:, index_fnzev])

    partition = [val >= 0 for val in eigenvectors[:, index_fnzev]]
    part_1 = [node for (node, nodeCommunity) in enumerate(partition) if nodeCommunity]
    part_2 = [node for (node, nodeCommunity) in enumerate(partition) if not nodeCommunity]
    cut_edges = []
    for edge in edges:
        v1, v2 = edge
        if v1 in part_1 and v2 in part_2 or v1 in part_2 and v2 in part_1:
            cut_edges.append(edge)

    new_edges = get_min_cuts(edges, cut_edges)
    new_nodes = get_nodes(new_edges)
    logging.debug('Remaining Edges: {} Cut Edges: {}'.format(len(new_edges), len(new_nodes)))

    cut_nodes = []
    for node in nodes:
        if node not in new_nodes:
            cut_nodes.append(node)

    new_part_1 = []
    new_part_2 = []
    for node in part_1:
        if node not in cut_nodes:
            new_part_1.append(node)
    for node in part_2:
        if node not in cut_nodes:
            new_part_2.append(node)

    logging.warning("Nodes in A: {} Nodes in B: {}".format(len(new_part_1), len(new_part_2)))
    logging.warning("Partition computed: nbA={} nbB={} (total {}), {} edges in between, {} cut nodes".format(
        len(part_1),
        len(part_2),
        len(nodes),
        len(cut_edges),
        len(cut_nodes),
    ))

    edges_part_1 = []
    edges_part_2 = []
    for edge in new_edges:
        if edge[0] in new_part_1:
            edges_part_1.append(edge)
        if edge[0] in new_part_2:
            edges_part_2.append(edge)

    with open(nodes_file + '_1.edg', 'w') as file_1:
        for edge in edges_part_1:
            file_1.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')

    with open(nodes_file + '_2.edg', 'w') as file_2:
        for edge in edges_part_2:
            file_2.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')

    return len(part_1), len(part_2)


def main():
    # Configure logging
    logging_format = '%(asctime)s.%(msecs)03d %(message)s'
    logging.basicConfig(format=logging_format, datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description="Compute the partition of a "
                                                 "graph using the Spectral Partition Algorithm.")

    parser.add_argument('--nodes-file', '-f', help='the file containing the nodes',
                        default='deadline.stg')
    parser.add_argument('--output-file', '-o', help='the filename of the'
                                                    ' communities PNG graph to be written')

    args = parser.parse_args()

    number_nodes, edges = partition_graph(args.nodes_file)


if __name__ == '__main__':
    main()
