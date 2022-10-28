import argparse

parser = argparse.ArgumentParser(description='Finds a solution to an instance of the Euclidean TSP via an evolutionary algorithm.')
parser.add_argument('--population', '-p', type=int, dest='population', help='The size of each generation of solutions', metavar='POPULATION', default=50)
parser.add_argument('--generations', '-g', type=int, dest='generations', help='The number of generations to iterate', metavar='GENERATIONS', default=50)
parser.add_argument('--cities', '-c', type=str, dest='filepath', help='The .csv file containing the names and coordinates of cities', metavar='FILEPATH')

def parse(argv):
    return parser.parse_args(argv[1:])