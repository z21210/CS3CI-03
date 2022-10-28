from random import shuffle

class GeneticPopulation:
    def __init__(this, initialiser, evaluator, offspringGenerator, offspringMutator, \
        populationSize=50):

        this.populationSize = populationSize
        this.initialiser = initialiser
        this.evaluator = evaluator
        this.offspringGenerator = offspringGenerator
        this.offspringMutator = offspringMutator

        this.population = [initialiser() for i in range(populationSize)]

    def selectParentsByTournament(this, population):
        parents = []
        shuffle(population)
        for i in range(0, len(population)-1, 2):
            if this.evaluator(population[i]) < this.evaluator(population[i+1]):
                parents.append(population[i])
            else:
                parents.append(population[i+1])
        return parents
    
    def generateOffspring(this, parents):
        offspring = []
        shuffle(parents)
        for i in range(0, len(parents)-1, 2):
            offspring += [*this.offspringGenerator(parents[i], parents[i+1])]
        return offspring

    def mutateOffspring(this, offspring):
        mutated = []
        for o in offspring:
            mutated.append(this.offspringMutator(o))
        return mutated
    
    def selectSurvivorsByThreshold(this, population):
        population.sort(key=this.evaluator)
        return population[:this.populationSize]

    def generateNextGeneration(this):
        this.population = this.selectSurvivorsByThreshold(
            this.population + this.offspringMutator(
                this.generateOffspring(
                    this.selectParentsByTournament(
                        this.population
                    )
                )
            )
        ) 
    
    def generateNextNGenerations(this, n):
        for i in range(n):
            this.generateNextGeneration()