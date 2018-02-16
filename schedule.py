import argparse

from partitioning import partition_graph


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

    partition_graph(args.nodes_file, True)
    partition_graph(args.nodes_file + '_1.edg', False)
    partition_graph(args.nodes_file + '_2.edg', False)
    part_1 = get_part_nodes(args.nodes_file + '_1.edg_1.edg')
    part_2 = get_part_nodes(args.nodes_file + '_1.edg_2.edg')
    part_3 = get_part_nodes(args.nodes_file + '_2.edg_1.edg')
    part_4 = get_part_nodes(args.nodes_file + '_2.edg_2.edg')
    print('part_1', part_1)
    print('part_2', part_2)
    print('part_3', part_3)
    print('part_4', part_4)


if __name__ == '__main__':
    main()
