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
#worstFitnessData = [None] * EVOLVE_ITERATIONS
#bestFitnessData = [None] * EVOLVE_ITERATIONS
#avgFitnessData = [None] * EVOLVE_ITERATIONS


worstFitnessData = numpy.empty( EVOLVE_ITERATIONS )
bestFitnessData = numpy.empty( EVOLVE_ITERATIONS )
avgFitnessData = numpy.empty( EVOLVE_ITERATIONS )

population = CreatePopulation(POPULATION_SIZE, NUMBER_OF_TRAITS, BOARD_SIZE_X, BOARD_SIZE_Y)

#run for desired evolution iterations
for j in range(0, EVOLVE_ITERATIONS ):
    pass

    #walk thru each individual in pop
    for i in range(0, POPULATION_SIZE):
        #store individual w/ their fitness data
        populationFitness[i] = PopulationFitness( population[i], EvalFitness(population[i]) )
       
    #display pop-fitness before sorting
    #print(populationFitness)

    #sort in ascending order by fitness (low/good to high/bad)
    populationFitness.sort(key=getFitness)

    #print(populationFitness)

    #copy sorted pop fitness data to reorder pop
    for i in range(0, POPULATION_SIZE):
        population[i] = populationFitness[i].individual

    worstFitnessData[j] = max( populationFitness, key=getFitness ).fitness
    bestFitnessData[j] = min( populationFitness, key=getFitness ).fitness

    #find avg
    fitnessSum = 0
    for i in range(0, POPULATION_SIZE):
        fitnessSum += populationFitness[i].fitness
    avgFitnessData[j] = int( fitnessSum/POPULATION_SIZE )

    #if first iteration 
    #if( j == 0 ):
        #select 2 parents from pop + show distr graph
    #    parents = BreedSelection(population, displayDistributionGraph=True)
    #else:
        #select 2 parents from pop
    parents = BreedSelection(population)

    #crossover breed parents to get children
    children = CrossoverBreed(parents[0], parents[1])

    #create possibly mutated children
    for child in children:
        #mutate child 
        Mutate(child)
        
    SurvivalReplacement(population, children)
    
    if( bestFitnessData[j] == 0 ):
        print("Best fitness of zero reached at iteration ", j)
    
t = numpy.arange(0, EVOLVE_ITERATIONS)

#plots:
plt.rcParams.update({'font.size': 22})
plt.figure(figsize = (10,15)) #1st arg = horizontal space, 2nd arg = vertical space
#plt.subplot(3, 1, 1)
plt.plot(t, bestFitnessData) 
plt.grid() #add a grid to graph
plt.title('Best Fitness per Iteration')
plt.ylabel('Best Fitness')
plt.xlabel('Iteration')
plt.show()

#plt.subplot(3, 1, 2)
plt.plot(t, avgFitnessData) 
plt.grid() #add a grid to graph
plt.title('Average Fitness per Iteration')
plt.ylabel('Average Fitness')
plt.xlabel('Iteration')
plt.show()

#plt.subplot(3, 1, 3)
plt.plot(t, worstFitnessData) 
plt.grid() #add a grid to graph
plt.title('Worst Fitness per Iteration')
plt.ylabel('Worst Fitness')
plt.xlabel('Iteration')
plt.show()