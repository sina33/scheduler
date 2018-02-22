import argparse

from partitioning import partition_graph, create_subgraph


def get_part_nodes(file):
    nodes = set()
    with open(file) as edges:
        for line in edges:
            a, b = line.split()
            nodes.add(int(a))
            nodes.add(int(b))
    return nodes


def main():
    parser = argparse.ArgumentParser(description="Compute the partition of a "
                                                 "graph using the Spectral Partition Algorithm.")

    parser.add_argument('--nodes-file', '-f', help='the file containing the nodes',
                        default='demo_nodes.txt')
    parser.add_argument('--output-file', '-o', help='the filename of the'
                                                    ' communities PNG graph to be written')

    args = parser.parse_args()

    part_1, part_2, nodes, edges, nodes_data, node_map = partition_graph(args.nodes_file)
    create_subgraph(part_1, edges, nodes_data, node_map, 'part_1.stg')
    create_subgraph(part_2, edges, nodes_data, node_map, 'part_2.stg')
    part_1, part_2, nodes, edges, nodes_data, node_map = partition_graph('part_1.stg')
    create_subgraph(part_1, edges, nodes_data, node_map, 'part_11.stg')
    create_subgraph(part_2, edges, nodes_data, node_map, 'part_12.stg')
    part_1, part_2, nodes, edges, nodes_data, node_map = partition_graph('part_2.stg')
    create_subgraph(part_1, edges, nodes_data, node_map, 'part_21.stg')
    create_subgraph(part_2, edges, nodes_data, node_map, 'part_22.stg')


if __name__ == '__main__':
    main()
