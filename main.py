import numpy as np
from random import shuffle, randint
import csv, sys, argparse
import math as maths
# # consider adding an nx plt of the best route so far
# import matplotlib.pyplot as plt
# import networkx as nx

parser = argparse.ArgumentParser(description='Finds a solution to an instance of the Euclidean TSP via an evolutionary algorithm.')
parser.add_argument('--population', '-p', type=int, dest='population', help='The size of each generation of solutions', metavar='POPULATION')
parser.add_argument('--generations', '-g', type=int, dest='generations', help='The number of generations to iterate', metavar='GENERATIONS')
args = parser.parse_args(sys.argv[1:])


def readCities(filename):
    """
    readCities Reads city names and coordinates from a .csv file with rows formatted as id, x, y.

    :param filename: The location of the .csv file
    :return: A list of dictionaries of city names and coordinates
    """
    cities = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append({
                'name': row['id'], 
                'x': float(row['x']), 
                'y': float(row['y'])
            })
    return cities

def getConnectivityMatrix(cities):
    """
    getConnectivityMatrix Generates a Euclidean connectivity matrix from a list of dictionaries of city names and coordinates.
    :param cities: The list of dictionaries containing city coordinates
    :return: A Euclidean connectivity matrix containing distances between cities
    """
    cityCount = len(cities)
    connectivityMatrix = np.zeros((cityCount, cityCount))
    for y in range(cityCount):
        for x in range(cityCount):
            fromCoords = (cities[y]['x'], cities[y]['y'])
            toCoords   = (cities[x]['x'], cities[x]['y'])
            connectivityMatrix[y,x] = getEuclideanDistance(fromCoords, toCoords)
    return connectivityMatrix

def getEuclideanDistance(c0, c1):
    """
    getEuclideanDistance Calculates the Euclidean distance between two coordinates.
    :param c0: A tuple containing the first pair of coordinates
    :param c1: A tuple containing the second pair of coordinates
    :return: The Euclidean distance between the two coordinates
    """
    return maths.sqrt((c1[0] - c0[0])**2 + (c1[1] - c0[1])**2)

def generateRoute(cities):
    """
    generateRoute Generates a random route through all of the cities, excluding the implicit start and end and the first city.
    :param cities: A list of cities
    :return: A list of indices of the cities in the route, excluding the implicit start and end index of 0
    """
    route = list(range(1,len(cities)))
    shuffle(route)
    return route

def evaluateRoute(m, route):
    """
    evaluateRoute Calculates the total distance of the route.
    :param m: A connectivity matrix containing the distances between cities
    :param route: A list of indices of cities in the route (start and end implied to be 0)
    :return: The total distance of the route
    """
    totalDistance = 0
    totalDistance += m[0, route[0]] # implied start from city 0
    for i in range(len(route)-1):
        totalDistance += m[route[i], route[i+1]]
    totalDistance += m[route[-1], 0] # implied return to city 0
    return totalDistance

def formatRoute(cities, route):
    """
    formatRoute Formats a route into a human readable string.
    :param cities: A list of dictionaries of city names
    :param route: A route through the cities (start and end implied to be 0)
    :return: The string representing the route
    """
    formatted = f"{cities[0]['name']}" # implied start from city 0
    for i in route:
        formatted += f" -> {cities[i]['name']}"
    formatted += f" -> {cities[0]['name']}" # implied return to city 0
    return formatted

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

def initialisePopulation(cities, n):
    """
    initialisePopulation Generates n random routes between cities.
    :param cities: A list of cities
    :param n: The size of the population to be generated
    :return: A list of n routes between the cities
    """
    return [generateRoute(cities) for i in range(n)]

def selectParentsByTournament(m, population):
    """
    selectParentsByTournament Pairs members of the population together, and returns the best of each pairing.
    :param m: The connectivity matrix of the cities
    :param population: The population from which to select parents
    :return: The survivors of the tournament
    """
    parents = []
    shuffle(population)
    for i in range(0, len(population)-1, 2):
        if evaluateRoute(m, population[i]) < evaluateRoute(m, population[i+1]):
            parents.append(population[i])
        else:
            parents.append(population[i+1])
    return parents

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
    

def generateOffspringByOrderOne(parents):
    """
    generateOffspringByOrderOne Generates offspring from a list of parents via the Order 1 algorithm.
    :param parents: The parents of this generation
    :return: A list of crossed over offspring
    """
    offspring = []
    shuffle(parents)
    for i in range(0, len(parents)-1, 2):
        offspring += [*orderOne(parents[i], parents[i+1])]
    return offspring

def randomTwoOpt(route):
    """
    randomTwoOpt Returns the result of executing a two-opt swap on a route between random edges.
    :param route: The route to perform a two-opt swap on
    :return: The result of performing a random two-opt swap on the route
    """
    i1 = randint(0, len(route)-1)
    i2 = randint(i1+1, len(route)+i1-1) % len(route)
    return twoOpt(route, i1, i2)

def mutatePopulationByTwoOpt(population):
    """
    mutatePopulationByTwoOpt Duplicates a population and performs a random two-opt swap on each duplicated route.
    :param population: The population to duplicate and mutate
    :return: The original population and mutated copy, appended
    """
    mutatedPopulation = []
    for route in population:
        mutatedPopulation.append(randomTwoOpt(route))
    return population + mutatedPopulation

def selectSurvivorsByThreshold(m, population, survivorsSize):
    """
    selectSurvivorsByThreshold Returns the best 'survivorSize' of a population.
    :param m: The connectivity matrix of cities
    :param population: The population to be selected from
    """
    population.sort(key= lambda p: evaluateRoute(m, p))
    return population[:survivorsSize]
    

def generateNextGeneration(m ,population, generationSize):
    """
    generateNextGeneration Generates a new generation of routes by selecting parents, generating and mutating offspring, then selecting survivors.
    :param m: The connectivity matrix of cities
    :param population: The last generation of routes
    """
    return selectSurvivorsByThreshold(
        m, 
        population + mutatePopulationByTwoOpt(
            generateOffspringByOrderOne(
                selectParentsByTournament(
                    m,
                    population
                )
            )
        ),
        generationSize
    )



cities = readCities('ulysses16.csv')
population = initialisePopulation(cities, args.population)
connections = getConnectivityMatrix(cities)
bestRoute, bestCost = None, None
for i in range(args.generations):
    population = generateNextGeneration(connections, population, args.population)
    best = population[0] # already sorted by cost
    print(f'{formatRoute(cities, best)}\nCost: {evaluateRoute(connections, best)}')
