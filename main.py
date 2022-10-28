from random import randint
import sys
from Population import GeneticPopulation
from TSP import TSP
from args import parse

def orderOne(parent1, parent2):
    """
    orderOne Generates two offspring by selecting the middle of each parent, and wrapping the other elements from the other parent.
    :param parent1: The first parent
    :param parent2: The second parent
    :return: Two offspring as a result of performing order one crossover
    """
    genomeLength = len(parent1)
    index1 = randint(0,genomeLength-1)
    index2 = (index1 + genomeLength//2) % genomeLength
    if index1 > index2:
        index1, index2 = index2, index1 # swap

    p1Start, p1Middle, p1End = parent1[0:index1], parent1[index1:index2], parent1[index2:]
    p2Start, p2Middle, p2End = parent2[0:index1], parent2[index1:index2], parent2[index2:]
    o1p1Sample, o2p2Sample = p1Middle, p2Middle
    o1p2Sample, o2p1Sample = p2End + [0] + p2Start + p2Middle, p1End + [0] + p1Start + p1Middle
    for e in o1p1Sample:
        o1p2Sample.remove(e)
    for e in o2p2Sample:
        o2p1Sample.remove(e)

    o1 = o1p1Sample + o1p2Sample
    o2 = o2p2Sample + o2p1Sample
    o1 = o1[o1.index(0)+1:] + o1[:o1.index(0)]
    o2 = o2[o2.index(0)+1:] + o2[:o2.index(0)]
    return o1, o2

def twoOpt(route, node1Index, node2Index):
    """
    twoOpt Performs a two-opt swap on the route and returns the result.
    :param route: A route represented by a list of indices
    :param node1Index: The index of the city at the begining of the swapped section
    :param node2Index: The index of the city at the end of the swapped section
    :return: The route produced by the two-opt swap 
    """
    if node2Index < node1Index:
        node1Index, node2Index = node2Index, node1Index # swap
    start = route[0:node1Index]
    middle = route[node1Index:node2Index+1]
    middle.reverse()
    end = route[node2Index+1:len(route)]
    return start + middle + end

def randomTwoOpt(route):
    """
    randomTwoOpt Returns the result of executing a two-opt swap on a route between random edges.
    :param route: The route to perform a two-opt swap on
    :return: The result of performing a random two-opt swap on the route
    """
    i1 = randint(0, len(route)-1)
    i2 = randint(i1+1, len(route)+i1-1) % len(route)
    return twoOpt(route, i1, i2)


args = parse(sys.argv)
tsp = TSP(args.filepath)
population = GeneticPopulation(
    initialiser=tsp.generateRandomRoute, 
    evaluator=tsp.evaluateRoute, 
    offspringGenerator=orderOne, 
    offspringMutator=randomTwoOpt,
    populationSize=args.population
)

bestRoute, bestCost = None, None
for i in range(args.generations):
    population.generateNextGeneration()
    best = population.population[0] # already sorted by cost
    print(f'{tsp.formatRoute(best)}\nCost: {tsp.evaluateRoute(best)}')
