#import all functs from 8 queens functs
from EightQueensFuncts import *
import matplotlib.pyplot as plt

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

population = CreatePopulation(POPULATION_SIZE, NUMBER_OF_TRAITS, BOARD_SIZE_X, BOARD_SIZE_Y)
print(population)

for individual in population:
    print(EvalFitness(individual))