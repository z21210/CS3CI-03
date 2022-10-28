import math as maths
from random import shuffle
import numpy as np
import csv

class TSP:
    def __init__(this, filename):
        this.cities = this.readCities(filename)
        this.connections = this.getConnectivityMatrix()

    def readCities(this, filename):
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

    def getConnectivityMatrix(this):
        """
        getConnectivityMatrix Generates a Euclidean connectivity matrix from a list of dictionaries of city names and coordinates.
        :return: A Euclidean connectivity matrix containing distances between cities
        """
        cityCount = len(this.cities)
        connectivityMatrix = np.zeros((cityCount, cityCount))
        for y in range(cityCount):
            for x in range(cityCount):
                fromCoords = (this.cities[y]['x'], this.cities[y]['y'])
                toCoords   = (this.cities[x]['x'], this.cities[x]['y'])
                connectivityMatrix[y,x] = this.getEuclideanDistance(fromCoords, toCoords)
        return connectivityMatrix

    def getEuclideanDistance(this, c0, c1):
        """
        getEuclideanDistance Calculates the Euclidean distance between two coordinates.
        :param c0: A tuple containing the first pair of coordinates
        :param c1: A tuple containing the second pair of coordinates
        :return: The Euclidean distance between the two coordinates
        """
        return maths.sqrt((c1[0] - c0[0])**2 + (c1[1] - c0[1])**2)

    def generateRandomRoute(this):
        """
        generateRandomRoute Generates a random route through all of the cities, excluding the implicit start and end and the first city.
        :return: A list of indices of the cities in the route, excluding the implicit start and end index of 0
        """
        route = list(range(1,len(this.cities)))
        shuffle(route)
        return route

    def evaluateRoute(this, route):
        """
        evaluateRoute Calculates the total distance of the route.
        :param route: A list of indices of cities in the route (start and end implied to be 0)
        :return: The total distance of the route
        """
        totalDistance = 0
        totalDistance += this.connections[0, route[0]] # implied start from city 0
        for i in range(len(route)-1):
            totalDistance += this.connections[route[i], route[i+1]]
        totalDistance += this.connections[route[-1], 0] # implied return to city 0
        return totalDistance

    def formatRoute(this, route):
        """
        formatRoute Formats a route into a human readable string.
        :param route: A route through the cities (start and end implied to be 0)
        :return: The string representing the route
        """
        formatted = f"{this.cities[0]['name']}" # implied start from city 0
        for i in route:
            formatted += f" -> {this.cities[i]['name']}"
        formatted += f" -> {this.cities[0]['name']}" # implied return to city 0
        return formatted