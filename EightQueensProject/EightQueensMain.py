#import all functs from 8 queens functs
from dataclasses import dataclass
from statistics import mean
from EightQueensFuncts import *
import matplotlib.pyplot as plt
import numpy

"""
Instructions:
Collect data on the worst, average, and best fitness within the population at each iteration.
Create visuals of the data and write a short paper detailing your EA and results.

"""

POPULATION_SIZE = 100
NUMBER_OF_TRAITS = 8
BOARD_SIZE_X = 8
BOARD_SIZE_Y = 8
EVOLVE_ITERATIONS = 1000
CHILDREN_PER_ITERATION = 2 #same as number of replacements per iteration

@dataclass 
class PopulationFitness:
    """
    Class to keep track of all individuals and their fitness (even if individual's die).
    """
    individual: numpy.array
    fitness: int
    
#populationFitness = numpy.array( [None] * POPULATION_SIZE)

#populationFitness = numpy.array([None] * (POPULATION_SIZE + (EVOLVE_ITERATIONS * CHILDREN_PER_ITERATION) ) )
populationFitness = [None] * POPULATION_SIZE
#popFitHistoryIndex = 0

population = CreatePopulation(POPULATION_SIZE, NUMBER_OF_TRAITS, BOARD_SIZE_X, BOARD_SIZE_Y)
#print(population)

#walk thru each individual in pop
for i in range(0, POPULATION_SIZE):
    #store individual w/ their fitness data
    populationFitness[i] = PopulationFitness( population[i], EvalFitness(population[i]) )

def getFitness( individual: PopulationFitness ) -> int:
    return individual.fitness

#sort in ascending order by fitness (low/good to high/bad)
populationFitness.sort(key=getFitness)

print(populationFitness)

#copy sorted pop fitness data to reorder pop
for i in range(0, POPULATION_SIZE):
    population[i] = populationFitness[i].individual

worstFitnessData = max( populationFitness, key=getFitness )
bestFitnessData = min( populationFitness, key=getFitness )
#avgFitnessData = mean ( populationFitness) #currently a float, should be an int?

#find avg
fitnessSum = 0
for i in range(0, POPULATION_SIZE):
    fitnessSum += populationFitness[i].fitness
avgFitness = fitnessSum/POPULATION_SIZE

#select 2 parents from pop
parents = BreedSelection(population)

#crossover breed parents to get children
children = CrossoverBreed(parents[0], parents[1])

#create mutated children
for child in children:
    Mutate(child)
    
SurvivalReplacement(population, child)