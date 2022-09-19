"""
Author: Seth Cram
Class: Evolutionary Computation - CS472/CS572
Project 2 - Real Valued Vector Rep and Island Models
Due Date: 9/30/2022
Instructions:
    You will implement a Generational genetic algorithm to optimize 6 functions described in the link. 
    You will also implement an island model for these genetic algorithms.

    For this project we will use 6 of the functions:
    1. Spherical
    2. Rosenbrock
    3. Rastrigin
    4. Schwefel (The second one)
    5. Ackley
    6. Griewangk

Project Requirements:
    Create a set of GA populations and define your Island Model. Decide what migration interval and migration size. 
    Demonstrate that your island model works by showing one GA perform selection and selecting an individual that migrated from another population. 
    Your GAs should all use a Generational model.

    Run the GAs while not using an island model, collect data on the populations noting the Best, Average, and Worst fitness.

    Run the GAs while using an island model, collect data on the populations noting the Best, Average, and Worst fitness.

    Write a paper similar to the papers in project 1, compare your results of GAs and GAs with island models.
"""

import time

#import all functs from 8 queens functs
from RealValue_IslandModels_Lib import *
import matplotlib.pyplot as plt
import numpy
    
#init space for arrays

populationFitness = [None] * POPULATION_SIZE

#populationHistory = numpy.empty( EVOLVE_ITERATIONS, dtype=numpy.ndarray )
#phList = numpy.empty( INDIVIDUALS_NUMBER_OF_TRAITS, dtype=numpy.ndarray )

#elapsedTimeToFindSol = numpy.empty( DESIRED_SOLUTIONS, dtype=float)
#elapsedTimeToFindSol = []

worstFitnessData = numpy.empty(GENERATIONS_PER_RUN, dtype=float )
bestFitnessData = numpy.empty( GENERATIONS_PER_RUN, dtype=float )
avgFitnessData = numpy.empty( GENERATIONS_PER_RUN, dtype=float )

SHOW_FITNESS_DATA = True
MAX_ATTEMPTS_PER_ALG = 1
PARENTS_SAVED_FOR_ELITISM = 2

assert PARENTS_SAVED_FOR_ELITISM % 2 == 0, "Need to save an even number of parents for elitism."
assert PARENTS_SAVED_FOR_ELITISM < POPULATION_SIZE, "Can't save more parents for elitism than individuals in the population."

#sol number
solNumber = 0

#loop thru each function and their bounds
for functionEnum, functionBounds in functionBoundsDict.items():
    
    #Sets cannot have two items with the same value.
    solutions = set()
 
    start_time = time.time()

    while( len(solutions) < POSSIBLE_SOLUTIONS ):
        runsToFindSol = 0
        
        #create new and scrap old evo comp rslts if sol not found
        while(True):
            runsToFindSol += 1
            
            #create new population
            population = CreatePopulation(
                functionBounds=functionBounds, 
                population_size=POPULATION_SIZE, 
                individuals_num_of_traits=INDIVIDUALS_NUMBER_OF_TRAITS
            )

            #run for desired generations
            for j in range(0, GENERATIONS_PER_RUN ):

                #walk thru each individual in pop
                for i in range(0, POPULATION_SIZE):
                    individual = population[i]
                    individualFitness = EvalFitness(functionEnum, individual)
                    
                    #store individual w/ their fitness data
                    populationFitness[i] = IndividualFitness( individual, individualFitness )
                    
                    #if added individual is a sol
                    if(individualFitness == 0):
                        solutions.add(tuple(individual))
                    
                #sort in ascending order by fitness (low/good to high/bad)
                populationFitness.sort(key=getFitness)

                #print(populationFitness)

                worstFitnessData[j] = max( populationFitness, key=getFitness ).fitness
                bestFitnessData[j] = min( populationFitness, key=getFitness ).fitness

                #find avg
                fitnessSum = 0
                for i in range(0, POPULATION_SIZE):
                    #take the fitness sum
                    fitnessSum += populationFitness[i].fitness            
                avgFitnessData[j] =  fitnessSum/POPULATION_SIZE 
                    
                popIndex = 0
                    
                #Create a whole new pop from prev pop as parents
                for k in range(0, int(POPULATION_SIZE/2)):
                    
                    #if less children than parents saved for elitism
                    if( k < PARENTS_SAVED_FOR_ELITISM/2):
                        #apply elitism for next 2 most fit parents
                        children = populationFitness[k].individual, populationFitness[k+1].individual
                    
                    #not applying elitism
                    else:
                        #find parents
                        parents = BreedSelection(populationFitness)

                        #crossover breed parents to get children
                        children = CrossoverBreed(parents[0], parents[1])

                        #walk thru children
                        for child in children:
                            #mutate child 
                            Mutate(functionBounds=functionBounds, child=child)
                    
                    #walk thru gen'd children
                    for child in children:
                        #add to new population (reuse old space)
                        population[popIndex] = child
                        
                        popIndex += 1
                        
                assert popIndex == POPULATION_SIZE, "Size of population was changed to {}.".format(popIndex)
                
                #print("asdfs %d" % j)
                
                #if( bestFitnessData[j] == 0 ):
                #    print("Best fitness of zero reached for configuration " + str( populationFitness ) )
            
            #document best fitness per run
            print(
                "Run " + str(runsToFindSol) 
                + " resulted in a best fitness of " 
                + str(bestFitnessData[GENERATIONS_PER_RUN-1])
                + " for {}".format(functionEnum)
            )
            
            #if zero fitness reached so sol found or max attempts per alg exceeded
            if( bestFitnessData[GENERATIONS_PER_RUN-1] == 0 
               or MAX_ATTEMPTS_PER_ALG <= runsToFindSol ):
                #print("it took " + str(runsToFindSol) + " runs to find a solution")
                #exit loop
                break
        
        #print("My program took", elapsedTimeToFindSol[k], "seconds to run")
        
        if(SHOW_FITNESS_DATA):
            t = numpy.arange(0, GENERATIONS_PER_RUN)
            
            plt.rcParams.update({'font.size': 22})
            plt.plot(t, bestFitnessData) 
            plt.grid() #add a grid to graph
            plt.title('Best Fitness per Iteration for {}'.format(functionEnum))
            plt.ylabel('Best Fitness')
            plt.xlabel('Iteration')
            plt.show()

            #plt.subplot(3, 1, 2)
            plt.plot(t, avgFitnessData) 
            plt.grid() #add a grid to graph
            plt.title('Average Fitness per Iteration for {}'.format(functionEnum))
            plt.ylabel('Average Fitness')
            plt.xlabel('Iteration')
            plt.show()

            #plt.subplot(3, 1, 3)
            plt.plot(t, worstFitnessData) 
            plt.grid() #add a grid to graph
            plt.title('Worst Fitness per Iteration for {}'.format(functionEnum))
            plt.ylabel('Worst Fitness')
            plt.xlabel('Iteration')
            plt.show()
            
        #if max attempts per alg exceeded
        if(MAX_ATTEMPTS_PER_ALG <= runsToFindSol):
            #stop using this alg
            break
            
        #elapsedTimeToFindSol[k] = time.time() - start_time
        #elapsedTimeToFindSol.append(time.time() - start_time)
        
        solNumber += 1

    print(
        "All " + str( len(solutions) ) + " solutions: " + str(solutions) + " found in " 
        + str( time.time() - start_time ) + " seconds for {}.".format(functionEnum)
    )

"""
#t1 = numpy.arange(0, DESIRED_SOLUTIONS)
t1 = numpy.arange(0, k)

#plots:
plt.rcParams.update({'font.size': 22})

plt.figure(figsize = (10,15)) #1st arg = horizontal space, 2nd arg = vertical space
#plt.subplot(3, 1, 1)
plt.plot(t1, elapsedTimeToFindSol) 
plt.grid() #add a grid to graph
plt.title('Elapsed Time per Solution')
plt.ylabel('Elapsed Time (s)')
plt.xlabel('Solution')
plt.show()
"""