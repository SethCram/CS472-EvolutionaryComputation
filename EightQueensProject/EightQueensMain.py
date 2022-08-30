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
    
populationFitness = numpy.array( [None] * POPULATION_SIZE)

populationFitnessHistory = numpy.array([None] * (POPULATION_SIZE + (EVOLVE_ITERATIONS * CHILDREN_PER_ITERATION) ) )
popFitHistoryIndex = 0

population = CreatePopulation(POPULATION_SIZE, NUMBER_OF_TRAITS, BOARD_SIZE_X, BOARD_SIZE_Y)
#print(population)

#walk thru each individual in pop
for i in range(0, POPULATION_SIZE):
    #eval fitness of each individual and store it
    populationFitness[i] = EvalFitness(population[i])
    
    #store pop + fitness data
    populationFitnessHistory[popFitHistoryIndex] = PopulationFitness( population[i], populationFitness[i] )
    popFitHistoryIndex += 1
    
#print(populationFitnessHistory)

worstFitness = max( populationFitness )
bestFitness = min( populationFitness )
avgFitness = mean ( populationFitness  ) #currently a float, should be an int?