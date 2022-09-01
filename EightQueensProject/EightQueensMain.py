#import all functs from 8 queens functs
from EightQueensFuncts import *
import matplotlib.pyplot as plt
import numpy

"""
Instructions:
Collect data on the worst, average, and best fitness within the population at each iteration.
Create visuals of the data and write a short paper detailing your EA and results.

"""

#init unchanging constants
POPULATION_SIZE = 100
NUMBER_OF_TRAITS = 8
BOARD_SIZE_X = 8
BOARD_SIZE_Y = 8
EVOLVE_ITERATIONS = 1000
CHILDREN_PER_ITERATION = 2 #same as number of replacements per iteration
    
#init space for arrays

populationFitness = [None] * POPULATION_SIZE
worstFitnessData = [None] * EVOLVE_ITERATIONS
bestFitnessData = [None] * EVOLVE_ITERATIONS
avgFitnessData = [None] * EVOLVE_ITERATIONS

"""throws a type error for some reason
populationFitness = numpy.empty( POPULATION_SIZE )
worstFitnessData = numpy.empty( EVOLVE_ITERATIONS )
bestFitnessData = numpy.empty( EVOLVE_ITERATIONS )
avgFitnessData = numpy.empty( EVOLVE_ITERATIONS )
"""

population = CreatePopulation(POPULATION_SIZE, NUMBER_OF_TRAITS, BOARD_SIZE_X, BOARD_SIZE_Y)

#run for desired evolution iterations
for j in range(0, EVOLVE_ITERATIONS ):
    pass

    #walk thru each individual in pop
    for i in range(0, POPULATION_SIZE):
        #store individual w/ their fitness data
        populationFitness[i] = PopulationFitness( population[i], EvalFitness(population[i]) )
       
    #print pop-fitness before sorting
    print(populationFitness)

    #sort in ascending order by fitness (low/good to high/bad)
    populationFitness.sort(key=getFitness)

    #print(populationFitness)

    #copy sorted pop fitness data to reorder pop
    for i in range(0, POPULATION_SIZE):
        population[i] = populationFitness[i].individual

    worstFitnessData[j] = max( populationFitness, key=getFitness )
    bestFitnessData[j] = min( populationFitness, key=getFitness )

    #find avg
    fitnessSum = 0
    for i in range(0, POPULATION_SIZE):
        fitnessSum += populationFitness[i].fitness
    avgFitnessData[j] = fitnessSum/POPULATION_SIZE

    #if first iteration 
    if( j == 0 ):
        #select 2 parents from pop + show distr graph
        parents = BreedSelection(population, displayDistributionGraph=True)
    else:
        #select 2 parents from pop
        parents = BreedSelection(population)

    #crossover breed parents to get children
    children = CrossoverBreed(parents[0], parents[1])

    #create possibly mutated children
    for child in children:
        #mutate child 
        Mutate(child)
        
    SurvivalReplacement(population, children)