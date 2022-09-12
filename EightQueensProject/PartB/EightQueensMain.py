"""
Author: Seth Cram
Class: Evolutionary Computation - CS472/CS572
Project 1 part b - 8 Queen puzzle
Instructions:
    Collect data on the worst, average, and best fitness within the population at each iteration.
    Create visuals of the data and write a short paper detailing your EA and results.
"""

import time
start_time = time.time()

#import all functs from 8 queens functs
from EightQueensFuncts import *
import matplotlib.pyplot as plt
import numpy

#init unchanging constants
POPULATION_SIZE = 100
NUMBER_OF_TRAITS = 8
BOARD_SIZE_X = 8
BOARD_SIZE_Y = 8
EVOLVE_ITERATIONS = 1000
CHILDREN_PER_ITERATION = 2 #same as number of replacements per iteration
POSSIBLE_SOLUTIONS = 92
    
#init space for arrays

populationFitness = [None] * POPULATION_SIZE


worstFitnessData = numpy.empty( EVOLVE_ITERATIONS, dtype=int )
bestFitnessData = numpy.empty( EVOLVE_ITERATIONS, dtype=int )
avgFitnessData = numpy.empty( EVOLVE_ITERATIONS, dtype=int )

runsToFindSol = 0

#region test eval of fitness
"""#init arr w/ None
individual = numpy.array( [None] * NUMBER_OF_TRAITS )
individual[0] = 5
individual[1] = 6
individual[2] = 7
individual[3] = 4
individual[4] = 1
individual[5] = 2
individual[6] = 3
individual[7] = 0
indivFitness = EvalFitness(individual)

individual[0] = 0
individual[1] = 0
individual[2] = 0
individual[3] = 0
individual[4] = 0
individual[5] = 0
individual[6] = 0
individual[7] = 0
indivFitness = EvalFitness(individual) #expecting 26 (queens behind other queens shouldn't be threatened)
"""
#endregion test eval of fitness

#create new and scrap old evo comp rslts if sol not found
while(True):
    runsToFindSol += 1
    
    #create new population
    population = CreatePopulation(POPULATION_SIZE, BOARD_SIZE_X, BOARD_SIZE_Y)
    
    #walk thru each individual in pop
    for i in range(0, POPULATION_SIZE):
        #store individual w/ their fitness data
        populationFitness[i] = IndividualFitness( population[i], EvalFitness(population[i]) )
        
    #sort in ascending order by fitness (low/good to high/bad)
    populationFitness.sort(key=getFitness)

    #run for desired evolution iterations
    for j in range(0, EVOLVE_ITERATIONS ):

        #print(populationFitness)

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
        #    parents = BreedSelection(populationFitness, displayDistributionGraph=True)
        #else:
            #select 2 parents from pop
        parents = BreedSelection(populationFitness)

        #crossover breed parents to get children
        children = CrossoverBreed(parents[0].individual, parents[1].individual)

        #create possibly mutated children
        for child in children:
            #mutate child 
            Mutate(child)
            
        SurvivalReplacement(populationFitness, children)
        
        #print("asdfs %d" % j)
        
        #if( bestFitnessData[j] == 0 ):
        #    print("Best fitness of zero reached for configuration " + str( populationFitness ) )
    
    print("run " + str(runsToFindSol) + " resulted in a best fitness of " + str(bestFitnessData[EVOLVE_ITERATIONS-1]))
    
    #if zero fitness reached so sol found
    if( bestFitnessData[EVOLVE_ITERATIONS-1] == 0 ):
        print("it took " + str(runsToFindSol) + " runs to find a solution")
        #exit loop
        break
        

print("My program took", time.time() - start_time, "seconds to run")

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