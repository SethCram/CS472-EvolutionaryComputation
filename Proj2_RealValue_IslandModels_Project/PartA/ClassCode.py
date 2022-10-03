from copy import copy
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
        self.genomeSize = 10
        self.crossoverRate = 0.8 #high crossover rate similar to elitism but not same
        self.MutationStdDev = mutationStdDev
        #self.crossoverType
        #self.selectionType
        self.CurrentGeneration = 0
        self.population = []
        self.fitness = []
        self.child1 = None
        self.child2 = None
        self.parent1 = None
        self.parent2 = None
    def initPopulation(self):
        self.population = [ [(round(numpy.random.random() * 1025, 0))/100 - 512 for i in range(self.genomeSize)] for i in range(self.PopulationSize) ]
        self.CalcFitness()
    def CalcFitness(self):
        for individual in self.population:
            fitness = 0.0
            for gene in individual:
                fitness += gene**2
            self.fitness.append(fitness)
    def selection(self):
        """
        Tournament style sel where tournament size is 3.
        Ensure parents are unique individuals. 
        """
        
        tourneySize = 3
        
        #uniform random sel of index w/o replacement (so 2 unique parents)
        candidates = numpy.random.choice(100, size=tourneySize * 2) 
        
        p1 = candidates[0]
        
        for i in range(1, tourneySize):
            if self.fitness[i] < self.fitness[p1]:
                p1 = candidates[i]
                
        p2 = candidates[tourneySize]
                
        for i in range(tourneySize+1, tourneySize * 2):
            if self.fitness[i] < self.fitness[p2]:
                p2 = candidates[i]
                
        self.parent1 = self.population[p1]
        self.parent2 = self.population[p2]
        
    def crossover(self):
        """
        Uniform crossover.
        """
        self.child1 = []
        self.child2 = []
        
        for i in range(self.genomeSize):
            if numpy.random.random() < 0.5: #numpy random faster
                #self.child1 = self.parent1.copy() #cant bc rets shallow copy
                self.child1 = copy.deepcopy(self.parent1)
                self.child2 = copy.deepcopy(self.parent2)
                return
        
        for i in range(self.genomeSize):
            if numpy.random.random() < self.crossoverRate: #numpy random faster
                self.child1.append(self.parent1[i])
                self.child2.append(self.parent2[i])
            else:
                self.child1.append(self.parent2[i])
                self.child2.append(self.parent1[i])
            
        
    def mutate(self):
        pass
    def findMigrants(self, migrantSize):
        pass
    def addMigrants(self):
        pass
    def runGen(self):
        pass
    
