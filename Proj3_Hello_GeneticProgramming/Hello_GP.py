from enum import Enum
import matplotlib.pyplot as plt
import numpy
import multiprocessing
import time
import scipy.stats as ss

class InitType(Enum):
    GROWTH = 0
    FULL = 1

class Individual():
    def __init__(self, initDepth: int, initType: InitType) -> None:
        
        self.fitness = 0

    def Mutate(self):
        pass
    
    def __str__(self):
        return f"{self.name}({self.age})"  

class Population():
    def __init__(self, populationSize: int, initDepth: int, initType: InitType, mutationStdDev: float) -> None:
        self.populationSize = populationSize
        self.population = [Individual(initDepth, initType) for _ in range(self.populationSize)]
        self.avgFitness = 0
        self.bestFitness = 0
        self.worstFitness = 0
        
        for i in range(self.population):
            self.population[i].fitness = self.EvaluateFitness(self.population[i]) 
        
    def EvaluateFitness(self, individual: Individual):
        return (x**2 + y - 11)**2 + (x + y**2 -7)**2
        
    def Crossover(self, parent1: Individual, parent2: Individual) -> tuple:
        pass
    
    def ParentSelection(self) -> tuple:
        xIndexRange, prob = self.SetupHalfNormIntDistr(self.populationSize, stdDev=30)
    
        #if overloaded to display distr graph
        if(False):
            #take randos using the calc'd prob and index range
            nums = numpy.random.choice(xIndexRange, size = 1000000, p = prob)
            #display distr histogram
            plt.rcParams.update({'font.size': 22})
            plt.hist(nums, bins = pop_size)
            plt.title("likelihood of each parent index being chosen")
            plt.ylabel("likelihood of being chosen")
            plt.xlabel("parent index")
            plt.show()
    
        parent1Index, parent2Index = int( numpy.random.choice(xIndexRange, size = 2, p = prob) )
        
        #make sure indices within array range
        assert parent1Index < self.populationSize and parent2Index < self.populationSize and type(parent1Index) == int and type(parent2Index) == int
    
        return self.population[parent1Index], self.population[parent2Index]
    
    def SetupHalfNormIntDistr(pop_size: int, stdDev: int) -> tuple:
        """
        The half normal integer distribution parent indices are drawn from.

        Returns:
            tuple: index range and probability funct
        """
        #take interval 1-100
        x = numpy.arange(1, pop_size+1) #bc upper bound is exclusive
        #store every number's +/-0.5
        xU, xL = x + 0.5, x - 0.5 
        #determine probability
        prob = ss.halfnorm.cdf(xU, scale = stdDev) - ss.halfnorm.cdf(xL, scale = stdDev) #scale represents inner quartiles
        prob = prob / prob.sum() # normalize the probabilities so their sum is 1
        #decr by 1 to find the index 0-99
        xIndexRange = x - 1
    
        return xIndexRange, prob
    
    def __str__(self):
        return f"{self.name}({self.age})"

class GP():
    def __init__(self, populationSize: int, initDepth: int, initType: InitType, mutationStdDev: float):
        self.populationSize = populationSize
        self.mutationStdDev = mutationStdDev
        #self.crossoverType
        #self.selectionType
        self.CurrentGeneration = 0
        self.population = Population(populationSize, initDepth, initType, mutationStdDev)
        self.avgFitness = []
        self.bestFitness = []
        self.worstFitness = []
            
    def runGen(self) -> Population:
        pass
    
    def __str__(self):
        return f"{self.benchmark_funct} with pop size ({self.popSize})"

if __name__ == '__main__': 
    NT = set(numpy.add, numpy.subtract, numpy.multiply, numpy.divide) #division by zero always yields zero in integer arithmetic.
    T = set()
    POPULATION_SIZE = 100
    INDIVIDUALS_NUMBER_OF_TRAITS = 10
    GENERATIONS_PER_RUN = 300  
    TRAIT_CHANGE_PERCENTAGE = 3
    PAIRS_OF_PARENTS_SAVED_FOR_ELITISM = 1
    NUMBER_OF_ISLANDS = 5
    MIGRATION_INTERVAL = 5
    PAIRS_OF_IMMIGRANTS = 3
    MAX_CHILDREN_KILLED = 10
    CROWDING_DIST_THRESH = 1
    CROWDING_NICHES = 30
    FITNESS_SHARING_RANGE = 1