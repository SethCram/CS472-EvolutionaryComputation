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

from telnetlib import GA
import time

#import all functs from 8 queens functs
from RealValue_IslandModels_Lib import *
import matplotlib.pyplot as plt
import numpy

#init unchanging constants
POPULATION_SIZE = 100
NUMBER_OF_TRAITS = 8
POSSIBLE_SOLUTIONS = 92
DESIRED_SOLUTIONS = 5
GENERATIONS_PER_RUN = 100
    
#init space for arrays

populationFitness = [None] * POPULATION_SIZE

#populationHistory = numpy.empty( EVOLVE_ITERATIONS, dtype=numpy.ndarray )
#phList = numpy.empty( NUMBER_OF_TRAITS, dtype=numpy.ndarray )

#elapsedTimeToFindSol = numpy.empty( DESIRED_SOLUTIONS, dtype=float)
#elapsedTimeToFindSol = []

worstFitnessData = numpy.empty(GENERATIONS_PER_RUN, dtype=float )
bestFitnessData = numpy.empty( GENERATIONS_PER_RUN, dtype=float )
avgFitnessData = numpy.empty( GENERATIONS_PER_RUN, dtype=float )

SHOW_FITNESS_DATA = False

#Sets cannot have two items with the same value.
solutions = set()

#region test eval of fitness
#init arr w/ None
individual = numpy.array( [None] * NUMBER_OF_TRAITS )
individual[0] = 5
individual[1] = 6
individual[2] = 7
individual[3] = 4
individual[4] = 1
individual[5] = 2
individual[6] = 3
individual[7] = 0
indivFitness = EvalFitness(functionToOptimize=GA_Functions.Spherical, individual=individual)

individual[0] = 0
individual[1] = 0
individual[2] = 0
individual[3] = 0
individual[4] = 0
individual[5] = 0
individual[6] = 0
individual[7] = 0
indivFitness = EvalFitness(individual) #expecting 26 (queens behind other queens shouldn't be threatened)

#endregion test eval of fitness

#sol number
k = 0

start_time = time.time()

#dictionary of funct-domain bounds pairings
functionBoundsDict = { 
    GA_Functions.Spherical: (-5.12, 5.12),
    GA_Functions.Rosenbrock: (-2.048, 2.048),
    GA_Functions.Rastrigin: (-5.12, 5.12),
    GA_Functions.Schwefel2: (-512.03, 511.97),
    GA_Functions.Ackley: (-30, 30),
    GA_Functions.Griewangk: (-600, 600)
}

#for k in range(0, DESIRED_SOLUTIONS):
while( len(solutions) < POSSIBLE_SOLUTIONS ):
    runsToFindSol = 0
    
    #create new and scrap old evo comp rslts if sol not found
    while(True):
        runsToFindSol += 1
        
        #create new population
        population = CreatePopulation(functionBounds=functionBoundsDict[GA_Functions.Spherical], population_size=100, individuals_num_of_traits=8)
        
        #walk thru each individual in pop
        for i in range(0, POPULATION_SIZE):
            individual = population[i]
            individualFitness = EvalFitness(individual)
            
            #store individual w/ their fitness data
            populationFitness[i] = IndividualFitness( individual, individualFitness )
            
            #if added individual is a sol
            if(individualFitness == 0):
                solutions.add(tuple(individual))
            
        #sort in ascending order by fitness (low/good to high/bad)
        populationFitness.sort(key=getFitness)

        #run for desired evolution iterations
        for j in range(0, GENERATIONS_PER_RUN ):

            #print(populationFitness)

            worstFitnessData[j] = max( populationFitness, key=getFitness ).fitness
            bestFitnessData[j] = min( populationFitness, key=getFitness ).fitness

            #find avg
            fitnessSum = 0
            for i in range(0, POPULATION_SIZE):
                #take the fitness sum
                fitnessSum += populationFitness[i].fitness            
            avgFitnessData[j] = int( fitnessSum/POPULATION_SIZE )
            
            #attempt at trait tracking
            
            #store pop every evolution iteration
            #populationHistory[j] = population
            #phList[:] = population
            
            #for i in range(0, NUMBER_OF_TRAITS):
                #phList[j][i] = []
                #copy over each individual's i'th col into this iteration's phList (each list is now a col instead of a row)
            #    phList[i] = [individual[i] for individual in populationHistory[j]]
                #bleh = [individual[i] for individual in populationHistory[j]]
            #populationHistory[j] = phList[j]
            
            #plt.hist(phList[0], bins = NUMBER_OF_TRAITS)
            #plt.show()

            #if first iteration 
            #if( j == 0 ):
                #select 2 parents from pop + show distr graph
            #    parents = BreedSelection(populationFitness, displayDistributionGraph=True)
            #else:
                #select 2 parents from pop
            parents = BreedSelection(populationFitness)

            #crossover breed parents to get children
            children = CrossoverBreed(parents[0].individual, parents[1].individual)

            #walk thru children
            for child in children:
                #mutate child 
                Mutate(child)
                
                #if fitness of child is 0
                if(EvalFitness(child) == 0):
                    #add child as a tuple to sols set
                    solutions.add(tuple(child))
                
            SurvivalReplacement(populationFitness, children)
            
            #print("asdfs %d" % j)
            
            #if( bestFitnessData[j] == 0 ):
            #    print("Best fitness of zero reached for configuration " + str( populationFitness ) )
        
        #print("run " + str(runsToFindSol) + " resulted in a best fitness of " + str(bestFitnessData[EVOLVE_ITERATIONS-1]))
        
        #if zero fitness reached so sol found
        if( bestFitnessData[GENERATIONS_PER_RUN-1] == 0 ):
            #print("it took " + str(runsToFindSol) + " runs to find a solution")
            #exit loop
            break
        
    #elapsedTimeToFindSol[k] = time.time() - start_time
    #elapsedTimeToFindSol.append(time.time() - start_time)
    
    k += 1
    
    #print("My program took", elapsedTimeToFindSol[k], "seconds to run")
    
    if(SHOW_FITNESS_DATA):
        t = numpy.arange(0, GENERATIONS_PER_RUN)
        
        plt.rcParams.update({'font.size': 22})
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

print("All " + str( len(solutions) ) + " solutions: " + str(solutions) + " found in " 
      + str( time.time() - start_time ) + " seconds.")

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