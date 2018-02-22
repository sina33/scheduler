import argparse
import logging
from collections import Counter

import numpy as np


def degree_nodes(adjacency_matrix, total_nodes):
    """
    Compute the degree of each node
    Returns the vector of degrees
    """

    d = []
    for i in range(total_nodes):
        d.append(sum([adjacency_matrix[i][j] for j in range(total_nodes)]))

    return d


def get_min_cuts(edges, cut_edges):
    logging.debug('total edges {} in between {}'.format(len(edges), len(cut_edges)))
    if len(cut_edges) == 0:
        return edges

    nodes = []
    for v1, v2 in cut_edges:
        nodes.append(v1)
        nodes.append(v2)
    new_edges = []
    node = Counter(nodes).most_common()[0][0]
    for v1, v2 in edges:
        if v1 != node and v2 != node:
            new_edges.append((v1, v2))
    new_edges_in_between = []
    for edge in cut_edges:
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


def get_cut_edges(edges, part_1, part_2):
    cut_edges = []
    for edge in edges:
        v1, v2 = edge
        if v1 in part_1 and v2 in part_2 or v1 in part_2 and v2 in part_1:
            cut_edges.append(edge)
    return cut_edges


def import_nodes_from_tg(tg_file):
    nodes = []
    edges = []
    node_map = {}

    with open(tg_file) as f:
        lines = f.readlines()
        total_nodes = int(lines[0])
        adjacency_matrix = np.zeros((total_nodes, total_nodes), dtype=int)
        for index in range(1, len(lines)):
            row = lines[index].split()
            node_name = row[0]
            node_id = index - 1
            node_map[node_name] = node_id
            nodes.append(node_id)

            for dep in row[4:]:
                dep_id = node_map[dep]
                edges.append((node_id, dep_id))
                adjacency_matrix[node_id][dep_id] = 1
                adjacency_matrix[dep_id][node_id] = 1

    logging.info("Imported {} nodes with {} edges from {}".format(total_nodes, len(edges), tg_file))
    return nodes, edges, adjacency_matrix, degree_nodes(adjacency_matrix, total_nodes), lines, node_map


def prune(part_1, part_2, nodes, new_nodes):
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
    return new_part_1, new_part_2, cut_nodes


def get_deps(node, edges, node_map):
    deps = []
    dep_names = []
    for edge in edges:
        v1, v2 = edge
        if v1 == node and v2 < v1:
            deps.append(v2)
        elif v2 == node and v1 < v2:
            deps.append(v1)
    for node in deps:
        dep_names.append(get_node_name(node, node_map))
    return dep_names


def get_node_name(node_id, node_map):
    for key, value in node_map.items():
        if str(value) == str(node_id):
            return str(key)
    # print("unable to find node {} in".format(node_id), node_map)
    return 'ERROR'


def create_subgraph(part, edges, nodes_data, node_map, name):
    with open(name, 'w') as file:
        flag = False
        if part[0] != 0:
            part.insert(0, 0)
            node_map['0'] = 0
            nodes_data.insert(1, '0      	 0      	 0      	 0')
            flag = True

        file.write(str(len(part)) + '\n')
        for node in part:
            node_data = nodes_data[node + 1].split()
            file.write(get_node_name(node, node_map) + '\t\t' + node_data[1] + '\t\t' + node_data[2] + '\t\t')
            deps = get_deps(node, edges, node_map)
            if flag and node != 0:
                if len(deps) != 0:
                    flag = False
                else:
                    deps = [0]
            file.write(str(len(deps)))
            for dep in deps:
                file.write('\t\t' + str(dep))
            file.write('\n')


def partition_graph(nodes_file):
    logging.basicConfig(level=logging.INFO)
    logging.info("Computing Adjacency Matrix...")

    nodes, edges, adjacency_matrix, degrees, nodes_data, node_map = import_nodes_from_tg(nodes_file)

    logging.info("Computing the Laplacian matrix...")
    laplacian_matrix = np.diag(degrees) - adjacency_matrix

    logging.info("Computing the eigenvectors and eigenvalues...")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # Index of the second eigenvalue
    index_fnzev = np.argsort(eigenvalues)[1]
    logging.debug("Eigenvector for #{} eigenvalue ({}): ".format(
        index_fnzev, eigenvalues[index_fnzev]), eigenvectors[:, index_fnzev])

    partition = [val >= 0 for val in eigenvectors[:, index_fnzev]]
    part_1 = [node for (node, nodeCommunity) in enumerate(partition) if nodeCommunity]
    part_2 = [node for (node, nodeCommunity) in enumerate(partition) if not nodeCommunity]
    cut_edges = get_cut_edges(edges, part_1, part_2)
    edges = get_min_cuts(edges, cut_edges)
    new_nodes = get_nodes(edges)
    logging.info("Nodes in A: {} Nodes in B: {}".format(len(part_1), len(part_2)))

    part_1, part_2, cut_nodes = prune(part_1, part_2, nodes, new_nodes)
    logging.info("Partition computed: nbA={} nbB={} (total {}), {} edges in between, {} cut nodes".format(
        len(part_1),
        len(part_2),
        len(nodes),
        len(cut_edges),
        len(cut_nodes),
    ))
    return part_1, part_2, nodes, edges, nodes_data, node_map


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

    partition_graph(args.nodes_file)


if __name__ == '__main__':
    main()
