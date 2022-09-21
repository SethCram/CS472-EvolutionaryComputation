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
import multiprocessing
    
#init space for arrays

populationFitness = [None] * Implementation_Consts.POPULATION_SIZE

#populationsArr = numpy.array( [None] * Implementation_Consts.NUMBER_OF_ISLANDS)

#populationHistory = numpy.empty( Implementation_Consts.EVOLVE_ITERATIONS, dtype=numpy.ndarray )
#phList = numpy.empty( Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, dtype=numpy.ndarray )

#elapsedTimeToFindSol = numpy.empty( DESIRED_SOLUTIONS, dtype=float)
#elapsedTimeToFindSol = []

worstFitnessData = numpy.empty(Implementation_Consts.GENERATIONS_PER_RUN, dtype=float )
bestFitnessData = numpy.empty( Implementation_Consts.GENERATIONS_PER_RUN, dtype=float )
avgFitnessData = numpy.empty( Implementation_Consts.GENERATIONS_PER_RUN, dtype=float )

SHOW_FITNESS_DATA = True
MAX_ATTEMPTS_PER_ALG = 1

USE_ISLAND_MODEL = False
RUN_IN_PARRALLEL = False

#sol number
solNumber = 0

#loop thru each function and their bounds
for functionEnum, functionBounds in functionBoundsDict.items():
    
    if(USE_ISLAND_MODEL):
        #if island model in parrallel
        if(RUN_IN_PARRALLEL):
            
            procArr = []
            
            """
            # creating a pipe
            parent_conn0, child_conn1 = multiprocessing.Pipe()
            
            p0 = multiprocessing.Process(
                target=RunIsland, 
                args=(
                    functionEnum, functionBounds, 
                    Implementation_Consts.POPULATION_SIZE,
                    Implementation_Consts.GENERATIONS_PER_RUN,
                    Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                    Implementation_Consts.PARENTS_SAVED_FOR_ELITISM,
                    USE_ISLAND_MODEL,
                    Implementation_Consts.MIGRATION_INTERVAL,
                    Implementation_Consts.MIGRATION_SIZE,
                    
                    SHOW_FITNESS_DATA,
                    # parent_conn,msgs
                )
            )
            
            #run parrallel islands w/ migration
            for i in range(0, int(Implementation_Consts.NUMBER_OF_ISLANDS/2) ):

                # creating a pipe
                parent_conn, child_conn = multiprocessing.Pipe()
                
                if( i == 0 ):
                    # creating new processes
                    p1 = multiprocessing.Process(
                        target=RunIsland, 
                        args=(
                            functionEnum, functionBounds, 
                            Implementation_Consts.POPULATION_SIZE,
                            Implementation_Consts.GENERATIONS_PER_RUN,
                            Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                            Implementation_Consts.PARENTS_SAVED_FOR_ELITISM,
                            USE_ISLAND_MODEL,
                            Implementation_Consts.MIGRATION_INTERVAL,
                            Implementation_Consts.MIGRATION_SIZE,
                            parent_conn, child_conn,
                            SHOW_FITNESS_DATA,
                            # parent_conn,msgs   
                        )
                    )
                    
                else:
                    # creating new processes
                    p1 = multiprocessing.Process(
                        target=RunIsland, 
                        args=(
                            functionEnum, functionBounds, 
                            Implementation_Consts.POPULATION_SIZE,
                            Implementation_Consts.GENERATIONS_PER_RUN,
                            Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                            Implementation_Consts.PARENTS_SAVED_FOR_ELITISM,
                            USE_ISLAND_MODEL,
                            Implementation_Consts.MIGRATION_INTERVAL,
                            Implementation_Consts.MIGRATION_SIZE,
                            parent_conn, child_conn,
                            SHOW_FITNESS_DATA,
                            # parent_conn,msgs
                        )
                    )
                p2 = multiprocessing.Process(
                    target=RunIsland, 
                    args=(child_conn,)
                )
                
                procArr.append(p1)
                procArr.append(p2)
            """
                
        #if island model but not in parrallel
        else:
            #run sequential islands w/ no migration
            for i in range(0, Implementation_Consts.NUMBER_OF_ISLANDS):
                RunIsland(
                    functionEnum, functionBounds, 
                    Implementation_Consts.POPULATION_SIZE,
                    Implementation_Consts.GENERATIONS_PER_RUN,
                    Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                    Implementation_Consts.PARENTS_SAVED_FOR_ELITISM,
                    USE_ISLAND_MODEL,
                    Implementation_Consts.MIGRATION_INTERVAL,
                    Implementation_Consts.MIGRATION_SIZE,
                    show_fitness_plots=SHOW_FITNESS_DATA,
                )
    #if not using island model
    else:
        RunIsland(
            functionEnum, functionBounds, 
            Implementation_Consts.POPULATION_SIZE,
            Implementation_Consts.GENERATIONS_PER_RUN,
            Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
            Implementation_Consts.PARENTS_SAVED_FOR_ELITISM,
            show_fitness_plots=SHOW_FITNESS_DATA
        )
    
    """
    #Sets cannot have two items with the same value.
    solutions = set()
 
    start_time = time.time()
    
#    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
    
    while( len(solutions) < Implementation_Consts.POSSIBLE_SOLUTIONS ):
        runsToFindSol = 0
        
        
        #create new and scrap old evo comp rslts if sol not found
        while(True):
            runsToFindSol += 1
            
            #create new population
            population = CreatePopulation(
                functionBounds=functionBounds, 
                population_size=Implementation_Consts.POPULATION_SIZE, 
                individuals_num_of_traits=Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS
            )
            
            #store pop in pops arr
            #populationsArr[i] = population

            #run for desired generations
            for j in range(0, Implementation_Consts.GENERATIONS_PER_RUN ):

                #walk thru each individual in pop
                for i in range(0, Implementation_Consts.POPULATION_SIZE):
                    individual = population[i]
                    individualFitness = EvalFitness(functionEnum, individual)
                    
                    #store individual w/ their fitness data
                    populationFitness[i] = IndividualFitness( individual, individualFitness )
                    
                    #if added individual is a sol
                    if(individualFitness == 0):
                        solutions.add(tuple(individual))
                    
                #sort in ascending order by fitness (low/good to high/bad)
                populationFitness.sort(key=getFitness)
                
                #if generation is on a migration interval
                if( j % (Implementation_Consts.MIGRATION_INTERVAL-1) == 0 ):
                    #take migrant sized section of most fit individuals
                    migrationPopFit = populationFitness[:Implementation_Consts.MIGRATION_SIZE]
                    
                    #pipe it over to the next population
                    
                    #pipe a keyword to get the listener to stop listening
                    
                    #listen for more pipe data until the stop listening keyword is seen
                    
                    #use the recieved migrants to replace the old ones

                #print(populationFitness)

                worstFitnessData[j] = max( populationFitness, key=getFitness ).fitness
                bestFitnessData[j] = min( populationFitness, key=getFitness ).fitness

                #find avg
                fitnessSum = 0
                for i in range(0, Implementation_Consts.POPULATION_SIZE):
                    #take the fitness sum
                    fitnessSum += populationFitness[i].fitness            
                avgFitnessData[j] =  fitnessSum/Implementation_Consts.POPULATION_SIZE 
                    
                popIndex = 0
                    
                #Create a whole new pop from prev pop as parents
                for k in range(0, int(Implementation_Consts.POPULATION_SIZE/2)):
                    
                    #if less children than parents saved for elitism
                    if( k < Implementation_Consts.PARENTS_SAVED_FOR_ELITISM/2):
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
                            Mutate(functionBounds=functionBounds, child=child, trait_change_percentage=Implementation_Consts.TRAIT_CHANGE_PERCENTAGE)
                    
                    #walk thru gen'd children
                    for child in children:
                        #add to new population (reuse old space)
                        population[popIndex] = child
                        
                        popIndex += 1
                        
                assert popIndex == Implementation_Consts.POPULATION_SIZE, "Size of population was changed to {}.".format(popIndex)
                
                #print("asdfs %d" % j)
                
                #if( bestFitnessData[j] == 0 ):
                #    print("Best fitness of zero reached for configuration " + str( populationFitness ) )
            
            #document best fitness per run
            print(
                "Run " + str(runsToFindSol) 
                + " resulted in a best fitness of " 
                + str(bestFitnessData[Implementation_Consts.GENERATIONS_PER_RUN-1])
                + " for {}".format(functionEnum)
            )
            
            #if zero fitness reached so sol found or max attempts per alg exceeded
            if( bestFitnessData[Implementation_Consts.GENERATIONS_PER_RUN-1] == 0 
            or MAX_ATTEMPTS_PER_ALG <= runsToFindSol ):
                #print("it took " + str(runsToFindSol) + " runs to find a solution")
                #exit loop
                break
        
        #print("My program took", elapsedTimeToFindSol[k], "seconds to run")
        
        if(SHOW_FITNESS_DATA):
            t = numpy.arange(0, Implementation_Consts.GENERATIONS_PER_RUN)
            
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