import numpy

#sphere has dom range of -5.12, 5.12
individual = [(round(numpy.random.random() * 1025, 0))/100 - 512 for i in range(10)]
individual

class GA:
    """
    Islands exist outside of GA.
    """
    def __init__(self, popSize, mutationStdDev):
        self.PopulationSize = popSize
        self.MutationStdDev = mutationStdDev
        #self.crossoverType
        #self.selectionType
        self.CurrentGeneration = 0
        self.population = []
        self.fitness = []
    def initPopulation(self):
        self.population = [ [(round(numpy.random.random() * 1025, 0))/100 - 512 for i in range(10)] for i in range(self.PopulationSize) ]
        self.CalcFitness()
    def CalcFitness(self):
        for individual in self.population:
            fitness = 0.0
            for gene in individual:
                fitness += gene**2
            self.fitness.append(fitness)
    def selection(self):
        pass
    def crossover(self):
        pass
    def mutate(self):
        pass
    def findMigrants(self, migrantSize):
        pass
    def addMigrants(self):
        pass
    def runGen(self):
        pass
    
