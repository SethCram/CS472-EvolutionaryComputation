"""
Author: Seth Cram
Class: Evolutionary Computation - CS472/CS572
Project 2b: Real Valued Vector Rep and Island Models, Fitness Sharing, and Crowding
Due Date: 10/10/2022
Instructions:
    Project 2 will consist of creating multiple Genetic Algorithms, and having them evolve in unison in order to find optimal solutions to a set of benchmark functions.

    You will implement a Generational genetic algorithm to optimize 6 functions described in the link. You will also implement an island model for these genetic algorithms.

    Your genome length should be 10.

    For this project we will use 6 of the functions:
    1. Spherical
    2. Rosenbrock
    3. Rastrigin
    4. Schwefel (The second one)
    5. Ackley
    6. Griewangk

Project Requirements:
    Create a set of GA's with the options to use: Island Models(IM), Fitness Sharing (FS), and Crowding (CR).

    Your GAs should all use a Generational model.

    You will run a GA using the following configurations:
    GA
    GA + IM
    GA + FS
    GA + CR
    GA + IM + FS (doesn't work)
    GA + IM + CR
    GA + FS + CR
    GA + IM + FS + CR (doesn't work)

    Write a paper similar to the papers in project 2, skip describing the problems, but go into detail describing your implementation.

    Do not describe your code structures, but describe the algorithm of your implementation. Plot your results.
"""

import time

#import all functs from 8 queens functs
from RealValue_IslandModels_Lib import *
import matplotlib.pyplot as plt
import numpy
import multiprocessing
    
SHOW_FITNESS_DATA = False

TEST_SEQ = False

#test settings
#island_model = False
fitness_sharing = False
crowding = False

#sol number
solNumber = 0

islands = numpy.empty(Implementation_Consts.NUMBER_OF_ISLANDS, dtype=tuple)

#guard in the main module to avoid creating subprocesses recursively in Windows.
if __name__ == '__main__': 
    #loop thru each function and their bounds
    for functionEnum, functionBounds in functionBoundsDict.items():
        
        #if TEST_SEQ, run islands local and seq
        if(TEST_SEQ):
            #run sequential islands w/ no migration
            for i in range(0, Implementation_Consts.NUMBER_OF_ISLANDS):
                #run an islands and cache its resultant fitness data
                islands[i] =  RunIsland(
                    functionEnum, functionBounds, 
                    Implementation_Consts.POPULATION_SIZE,
                    Implementation_Consts.GENERATIONS_PER_RUN,
                    Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                    Implementation_Consts.PAIRS_OF_PARENTS_SAVED_FOR_ELITISM,
                    show_fitness_plots=SHOW_FITNESS_DATA, crowding=crowding, fitness_sharing=fitness_sharing
                )
                
            #store traits of best island for plotting
            bestFitness, bestFitnessData, avgFitnessData, worstFitnessData = FindBestIsland(islands)
                
            print("Best island's fitness is {}.".format(bestFitness) )
                
            t = numpy.arange(0, Implementation_Consts.GENERATIONS_PER_RUN)
                
            plt.title('Best {} Fitness Data'.format(functionEnum))
            displayStr = f'Best {functionEnum} Fitness Data'
            
            #plt.title('Best {}' +  + ' Fitness Data'.format(functionEnum))
            plt.title(displayStr)
            
            plt.plot(t, worstFitnessData, label='Worst Fitness') 
            plt.grid() #add a grid to graph
    
            plt.plot(t, avgFitnessData, label='Average Fitness') 
            plt.grid() #add a grid to graph
            
            plt.plot(t, bestFitnessData, label='Best Fitness') 
            plt.grid() #add a grid to graph
            
            plt.legend()
            plt.ylabel('Fitness')
            plt.xlabel('Generation')
            
            for fitnessData in (bestFitnessData, avgFitnessData, worstFitnessData):
                plt.annotate('%0.7f' % fitnessData.min(), xy=(1, fitnessData.max()), xytext=(8, 0), 
                            xycoords=('axes fraction', 'data'), textcoords='offset points')
            
            plt.show()
                
        #not TEST_SEQ, so auto set vars and run in parrallel
        else:
            num_of_configs = len(iterationVarConfig)
            
            #walk thru iteration var configs
            for configIndex in range(num_of_configs):
            
                #set config vars
                island_model, fitness_sharing, crowding, plotStr, lineTypeStr = iterationVarConfig[configIndex]
            
                #if island model in parrallel
                if(island_model):
                    
                    #init multi proccing queue
                    q = multiprocessing.Manager().Queue() 
                    
                    #parrallel plots won't show fitness plots (sometimes)
                    
                    procArr = numpy.empty(Implementation_Consts.NUMBER_OF_ISLANDS, dtype=multiprocessing.Process)
                    listenerPipeEnds = []
                    senderPipeEnds = []
                    
                    #walk thru islands
                    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
                        #init a pipe for each 
                        senderPipePart, listenerPipePart = multiprocessing.Pipe()
                        
                        #store the ends in separate lists
                        senderPipeEnds.append(senderPipePart)
                        listenerPipeEnds.append(listenerPipePart)
                    
                    #walk thru islands
                    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
                        
                        #special case for start
                        if( i == 0 ):
                            # creating new proc + add to proc arr
                            procArr[i] =  multiprocessing.Process(
                                target=RunIsland, 
                                args=(
                                    functionEnum, functionBounds, 
                                    Implementation_Consts.POPULATION_SIZE,
                                    Implementation_Consts.GENERATIONS_PER_RUN,
                                    Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                                    Implementation_Consts.PAIRS_OF_PARENTS_SAVED_FOR_ELITISM,
                                    island_model,
                                    Implementation_Consts.MIGRATION_INTERVAL,
                                    Implementation_Consts.PAIRS_OF_IMMIGRANTS,
                                    senderPipeEnds[i], #send to curr pipe index
                                    listenerPipeEnds[Implementation_Consts.NUMBER_OF_ISLANDS-1], #listen to prev pipe index w/ wrap around
                                    q,
                                    SHOW_FITNESS_DATA, crowding, fitness_sharing
                                )
                            )
                        #if not start
                        else:
                            #create new proc + add to proc arr
                            procArr[i] =  multiprocessing.Process(
                                target=RunIsland, 
                                args=(
                                    functionEnum, functionBounds, 
                                    Implementation_Consts.POPULATION_SIZE,
                                    Implementation_Consts.GENERATIONS_PER_RUN,
                                    Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                                    Implementation_Consts.PAIRS_OF_PARENTS_SAVED_FOR_ELITISM,
                                    island_model,
                                    Implementation_Consts.MIGRATION_INTERVAL,
                                    Implementation_Consts.PAIRS_OF_IMMIGRANTS,
                                    senderPipeEnds[i], #send to curr index pipe
                                    listenerPipeEnds[i-1], #listen to prev index pipe
                                    q,
                                    SHOW_FITNESS_DATA, crowding, fitness_sharing
                                )
                            )
                        
                    
                    #walk thru islands
                    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
                        #start each proc
                        procArr[i].start()
                    
                    #walk thru islands
                    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
                        #wait till each proc finished
                        procArr[i].join()
                    
                    islandsIndex = 0
                    
                    while not q.empty():
                            islands[islandsIndex] = q.get()
                            islandsIndex += 1
                #if parrallel non-migrating islands
                else:
                    #init multi proccing queue
                    q = multiprocessing.Manager().Queue() 
                    
                    procArr = numpy.empty(Implementation_Consts.NUMBER_OF_ISLANDS, dtype=multiprocessing.Process)
                    
                    #walk thru islands
                    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
                        # creating new proc + add to proc arr
                        procArr[i] =  multiprocessing.Process(
                            target=RunIsland, 
                            args=(
                                functionEnum, functionBounds, 
                                Implementation_Consts.POPULATION_SIZE,
                                Implementation_Consts.GENERATIONS_PER_RUN,
                                Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                                Implementation_Consts.PAIRS_OF_PARENTS_SAVED_FOR_ELITISM,
                                island_model,
                                Implementation_Consts.MIGRATION_INTERVAL,
                                Implementation_Consts.PAIRS_OF_IMMIGRANTS,
                                None, #no migration so don't need send and recieving pipe ends
                                None, 
                                q,
                                SHOW_FITNESS_DATA, crowding, fitness_sharing
                            )
                        )
                    
                    #walk thru islands
                    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
                        #start each proc
                        procArr[i].start()
                    
                    #walk thru islands
                    for i in range(Implementation_Consts.NUMBER_OF_ISLANDS):
                        #wait till each proc finished
                        procArr[i].join()
                    
                    islandsIndex = 0
                    
                    while not q.empty():
                            islands[islandsIndex] = q.get()
                            islandsIndex += 1
                
                
                #store traits of best island for plotting
                bestFitness, bestFitnessData, avgFitnessData, worstFitnessData = FindBestIsland(islands)
                    
                print("Best island's fitness is {}.".format(bestFitness) )
                
                #plot fitness data of best island
            
                t = numpy.arange(0, Implementation_Consts.GENERATIONS_PER_RUN)
                
                plt.rcParams.update({'font.size': 22})
                
                displayStr = f'Best {functionEnum} Fitness Data'
                    
                #plt.title('Best {}' +  + ' Fitness Data'.format(functionEnum))
                plt.title(displayStr)
                
                plt.plot(t, worstFitnessData, 'b' + lineTypeStr, label= 'Worst Fitness' + plotStr) 
                plt.grid() #add a grid to graph
        
                plt.plot(t, avgFitnessData, 'g' + lineTypeStr, label='Average Fitness' + plotStr) 
                plt.grid() #add a grid to graph
                
                plt.plot(t, bestFitnessData, 'r' + lineTypeStr, label='Best Fitness' + plotStr) 
                plt.grid() #add a grid to graph
                
                plt.legend()
                plt.ylabel('Fitness')
                plt.xlabel('Generation')
                
                fitnessIndex = 0
                
                textSize = 8
                
                if(configIndex == 0):
                    worstWorstFitness = worstFitnessData.max()
                
                for fitnessData in (worstFitnessData, avgFitnessData, bestFitnessData):
                    fitnessIndex += 1
                    
                    plt.annotate('%0.7f' % fitnessData.min(), xy=(1, worstWorstFitness + textSize/1.2 - configIndex * textSize * 3.5  - fitnessIndex * textSize * 1.2  ), xytext=(textSize, 0), 
                                xycoords=('axes fraction', 'data'), textcoords='offset points')#, verticalalignment='top')
                
                plt.show()
                
                #if last config plotted
                if( 
                   configIndex == num_of_configs-1 or 
                   configIndex == int( (num_of_configs-1) / 2 )
                ): 
                    #show combo plot  
                    plt.show()