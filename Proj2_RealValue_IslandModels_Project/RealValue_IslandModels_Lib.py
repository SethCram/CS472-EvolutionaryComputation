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

from enum import Enum
import random
import time
import numpy
import scipy.stats as ss
import matplotlib.pyplot as plt
from dataclasses import dataclass


class Implementation_Consts():
    """
    Unchanging constants useable, but not required, to use this library.
    Tests for this library leverage these constants.
    """
    #init unchanging constants
    POPULATION_SIZE = 100
    INDIVIDUALS_NUMBER_OF_TRAITS = 10
    POSSIBLE_SOLUTIONS = 1
    GENERATIONS_PER_RUN = 1000  #100: best fit = 0.583 #1000: best fit = 0.27 #10,000: best fit = 0.448??
    TRAIT_CHANGE_PERCENTAGE = 3
    
    PARENTS_SAVED_FOR_ELITISM = 2
    assert PARENTS_SAVED_FOR_ELITISM % 2 == 0, "Need to save an even number of parents for elitism."
    assert PARENTS_SAVED_FOR_ELITISM < POPULATION_SIZE, "Can't save more parents for elitism than individuals in the population."

    NUMBER_OF_ISLANDS = 3
    MIGRATION_INTERVAL = 5
    MIGRATION_SIZE = 6
    assert MIGRATION_SIZE % 2 == 0, "Need to save an even number of migrants for new generation."
    assert MIGRATION_SIZE < POPULATION_SIZE, "Can't select more migrants than individuals in the population."


#region GA enum and dicts
    
class GA_Functions(Enum):
    """
    A Python enum to represent what GA function is being optimized.
    """
    Spherical = 0
    Rosenbrock = 1
    Rastrigin = 2
    Schwefel2 = 3
    Ackley = 4
    Griewangk = 5
    
#dictionary of funct-domain bounds pairings
functionBoundsDict = { 
    GA_Functions.Spherical: (-5.12, 5.12),
    GA_Functions.Rosenbrock: (-2.048, 2.048),
    GA_Functions.Rastrigin: (-5.12, 5.12),
    GA_Functions.Schwefel2: (-512.03, 511.97),
    GA_Functions.Ackley: (-30, 30),
    GA_Functions.Griewangk: (-600, 600) 
}

#dictionary of funct-input target pairings
functionInputTargetDict = {
    GA_Functions.Spherical:  0,
    GA_Functions.Rosenbrock: 1,
    GA_Functions.Rastrigin: 0,
    GA_Functions.Schwefel2: -420.9687,
    GA_Functions.Ackley: 0,
    GA_Functions.Griewangk: 0 
}

#endregion GA enum and dicts

#region Fitness Functs and Class

def SqrdSum( inputArr: numpy.ndarray ) -> float:
    """
    Returns the elly-wise squared sum of the arr.
    """
    return sum(inputArr**2)

def CosOfTwoPiTrait( trait: float) -> float:
    return numpy.cos(2*numpy.pi*trait)

def EvalFitness( functionToOptimize: GA_Functions , individual: numpy.ndarray ) -> float:
    """
    Evaluates fitness of a single individual according to the GA function passed in.
    Optimal results value is 0. Further from a 0 result means higher fitness.
    """
    
    num_of_traits = len( individual )
    
    if( functionToOptimize == GA_Functions.Spherical):
        #walk thru each trait in the individual
        #for trait in individual:
        
        rslt = SqrdSum(individual)
        
    elif( functionToOptimize == GA_Functions.Rosenbrock):
        
        rslt = 0
        
        #leave out last elly
        for i in range(0,num_of_traits-1):
            #cache curr trait
            currTrait = individual[i]
            #calc new rslt and add to old one
            rslt += ( 100 * ( (individual[i+1] - currTrait**2)**2 ) + (currTrait-1)**2 )
    
    elif( functionToOptimize == GA_Functions.Rastrigin):
        #init w/ added val outside of summation
        rslt = 10 * num_of_traits
        
        for i in range(0,num_of_traits):
            #cache curr trait
            currTrait = individual[i]
            #calc new rslt and add to old one
            rslt += ( currTrait**2 - 10*CosOfTwoPiTrait(currTrait) )
        
        #make sure fitness is positive
        rslt = abs(rslt)
        
    elif( functionToOptimize == GA_Functions.Schwefel2):
        #init w/ added val outside of summation
        rslt = 418.9829 * num_of_traits
        
        for i in range(0,num_of_traits):
            #cache curr trait
            currTrait = individual[i]
            #calc new rslt and add to old one
            rslt += round( currTrait*numpy.sin(numpy.sqrt(abs(currTrait))), 4 )
        
        #make sure fitness is positive
        rslt = abs(rslt)
        
    elif( functionToOptimize == GA_Functions.Ackley):
        
        cosOfTwoPiTraitSum = 0
        
        for i in range(0,num_of_traits):
            #cache curr trait
            currTrait = individual[i]
            #take summation of a custom funct
            cosOfTwoPiTraitSum += round( CosOfTwoPiTrait(currTrait), 4)
            
        secondExpArg = -0.2*numpy.sqrt((1/num_of_traits)*SqrdSum(individual))
        thirdExpArg = (1/num_of_traits)*cosOfTwoPiTraitSum
            
        firstExp = numpy.e 
        secondExp = numpy.exp( secondExpArg )
        thirdExp = numpy.exp( thirdExpArg ) 
            
        #need to round the rslt to the 15th decimal place or else all 0 input rslts in near zero
        rslt = round( 20 + firstExp - 20*secondExp - thirdExp, 15)
        
        #make sure fitness is positive
        rslt = abs(rslt)
        
    elif( functionToOptimize == GA_Functions.Griewangk):
        rslt = 1
        
        rslt += sum( (individual**2) / 4000 )
        
        prodOfCosines = 1
        
        #go 1 to num_of_traits inclusive
        for i in range(1,num_of_traits+1):
            #cache curr trait
            currTrait = individual[i-1]
            
            #take the product over the individual's traits
            prodOfCosines *= numpy.cos(currTrait/numpy.sqrt(i))
        
        rslt -= prodOfCosines    
        
        #make sure fitness is positive
        rslt = abs(rslt)
    
    #desired val is 0 so should round reals down to 0 if close enough?
    
    
    #fitness should always be positive             
    return rslt

@dataclass 
class IndividualFitness:
    """
    Class to keep track of an individual and their fitness score.
    """
    individual: numpy.ndarray
    fitness: int

def getFitness( individual: IndividualFitness ) -> int:
        return individual.fitness
    
#endregion Fitness Functs and Class

#region Creation Functs

def CreateRandomIndividual( num_of_traits: int, lower_bound_inclusive: float, upper_bound_inclusive: float) -> numpy.ndarray:
    """
    Args:
        num_of_traits (int): Creates an individual with passed in number of random traits.
        lower_bound_inclusive (float): Used with a uniform distribution to randomly generate genes as floats.
        upper_bound_inclusive (float): Used with a uniform distribution to randomly generate genes as floats.

    Returns:
        numpy.ndarray: Individual containing the requested number of genes inclusively between the specified ranges.
    """
    
    #init arr w/ enough space
    individual = numpy.empty( num_of_traits, dtype=float )
    
    #walk thru each trait of individual
    for traitIndex in range(0, num_of_traits):
        #find new trait using random coords within bounds
        newTrait = random.uniform(lower_bound_inclusive, upper_bound_inclusive)
        
        #fill that individual's trait
        individual[traitIndex] = newTrait
        
    return individual

def CreatePopulation( functionBounds: tuple, population_size: int, individuals_num_of_traits: int) -> numpy.ndarray:
    """
    Args:
        functionToOptimize (GA_Function): Used to determine domain of input data.
        population_size (int): Determines how many random individuals created.
        individuals_num_of_traits (int): Tells how many traits each individual should have.

    Returns:
        numpy.ndarray: Population created.
    """
    
    #init pop
    population = numpy.array( [None] * population_size )
    
    lower_bound, upper_bound = functionBounds
    
    #populate every member of population
    for individualIndex in range(0, population_size):
        population[individualIndex] = CreateRandomIndividual( individuals_num_of_traits, lower_bound, upper_bound)
    
    return population

def CreateLocalPopulationFitness(functionEnum: GA_Functions, population: numpy.ndarray, solutions: set) -> list:
    """
    Creates local population fitness pairings array.
    Adds to the Solutions set if a solution is encountered during fitness evaluation.

    Args:
        functionEnum (GA_Functions): For fitness evaluation
        solutions (set): Pre-existing solutions

    Returns:
        list: local population fitness pairings array
    """
    local_pop_size = len(population)
    
    localPopulationFitness = [None] * local_pop_size
    
    #walk thru each individual in local pop
    for i in range(0, local_pop_size):
        individual = population[i]
        individualFitness = EvalFitness(functionEnum, individual)
        
        #store individual w/ their fitness data
        # don't need diff case for par island model as long as migration size 0 by default
        #populationFitness[popFitnessIndex] = IndividualFitness( individual, individualFitness )
        #popFitnessIndex += 1
        localPopulationFitness[i] = IndividualFitness( individual, individualFitness )
        
        #if added individual is a sol
        if(individualFitness == 0):
            solutions.add(tuple(individual))
    
    return localPopulationFitness

#endregion Creation Functs


def SetupHalfNormIntDistr(pop_size: int, stdDev: int) -> tuple:
    """
    The half normal integer distribution parent indices are drawn from.

    Returns:
        tuple: index range and probability funct
    """
    #take interval 1-100
    x = numpy.arange(1, pop_size+1) #bc upper bound is exclusive
    #store every number's +/-0.5
    xU, xL = x + 0.5, x - 0.5 
    #determine probability
    prob = ss.halfnorm.cdf(xU, scale = stdDev) - ss.halfnorm.cdf(xL, scale = stdDev) #scale represents inner quartiles
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    #decr by 1 to find the index 0-99
    xIndexRange = x - 1
    
    return xIndexRange, prob

def CrossoverBreed( parent1: numpy.ndarray, parent2: numpy.ndarray ) -> numpy.ndarray:
    """
    A 1-point crossover.
    Assumes both parents have the same number of traits.
    Returns two children contained in a numpy array.
    """
    
    num_of_traits = len(parent1)
    
    #init child arrs
    child1 = numpy.empty( num_of_traits, dtype=float ) #need to be below 0 or above 7
    child2 = numpy.empty( num_of_traits, dtype=float )
    children = numpy.array( [None] * 2 )
    
    #crossover point
    xpoint = random.randrange(1,num_of_traits-1) #don't want at very beginning or end bc don't wanna copy parents
    
    #take 1 point crossover
    child1 = numpy.concatenate( (parent1[:xpoint], parent2[xpoint:]) )
    child2 = numpy.concatenate( (parent2[:xpoint], parent1[xpoint:]) )
    #place childs into children arr
    children[0] = child1
    children[1] = child2

    return children

#region Selection Functs

def ParentSelection( populationFitness: list, displayDistributionGraph = False ) -> numpy.ndarray:
    """
    Assumes population array is sorted in ascending fitness order (low/good to high/bad).
    Returns an array of two parents.
    If displayDistributionGraph is True, random distr shown using random sample data.
    """
    
    #store pop size
    pop_size = len(populationFitness)
    
    #Using half norm and incr'd to take interval 1-100 then subtr 1 after.
    
    xIndexRange, prob = SetupHalfNormIntDistr(pop_size, stdDev=30)
    
    #if overloaded to display distr graph
    if( displayDistributionGraph):
        #take randos using the calc'd prob and index range
        nums = numpy.random.choice(xIndexRange, size = 1000000, p = prob)
        #display distr histogram
        plt.rcParams.update({'font.size': 22})
        plt.hist(nums, bins = pop_size)
        plt.title("likelihood of each parent index being chosen")
        plt.ylabel("likelihood of being chosen")
        plt.xlabel("parent index")
        plt.show()
        
    #choose parent indices, make sure only take int part of ret'd data
    parent1Index = int( numpy.random.choice(xIndexRange, size = 1, p = prob) )
    parent2Index = int( numpy.random.choice(xIndexRange, size = 1, p = prob) )
    
    #make sure indices within array range
    assert parent1Index < pop_size and parent2Index < pop_size and type(parent1Index) == int and type(parent2Index) == int
    
    #while parents are the same
    while( parent2Index == parent1Index):
        #refind parent 2
        parent2Index = int( numpy.random.choice(xIndexRange, size = 1, p = prob) )
    
    #get parents using indices
    parent1 = populationFitness[parent1Index]
    parent2 = populationFitness[parent2Index]
    #store parents in an array
    parentsArray = numpy.array( [None] * 2)
    parentsArray[0] = parent1.individual
    parentsArray[1] = parent2.individual
        
    #return chosen parents
    return parentsArray

def ImmigrantSelection(populationFitness: numpy.ndarray, desiredImmigrants: int) -> list:
    """
    Returns the desired number of immigrants using a fitness proportionate selection.  
    provideD that the passed in populationFitness is already sorted in ascending order.

    Args:
        populationFitness (numpy.ndarray): sorted in ascending order

    Returns:
        list: IndividualFitness pairings
    """
    
    #store pop size
    pop_size = len(populationFitness)
    
    xIndexRange, prob = SetupHalfNormIntDistr(pop_size, stdDev=30)
    
    #randomly select immigrant indices
    immigrantIndices = numpy.random.choice(xIndexRange, size = desiredImmigrants-1, p = prob)
    
    immigrants = [None] * desiredImmigrants
    
    i = 0
    for immigrantIndex in immigrantIndices:
        #make sure indices within array range
        assert immigrantIndex < pop_size
        
        #copy over into immigrants arr
        immigrants[i] = populationFitness[int(immigrantIndex)]
        i += 1    
    
    return immigrants

def FindBestIsland(islands: numpy.ndarray) -> tuple:
    """
    Determines the best island through comparing each island's best fit individual.

    Returns:
        tuple: bestFitness, bestFitnessData, avgFitnessData, worstFitnessData
    """
    num_of_islands = len(islands)
    
    #init best fitness w/ island 0's best fitness
    bestFitIslandIndex = 0
    bestFitness = islands[bestFitIslandIndex][0]
    
    #run thru islands
    for i in range(0, num_of_islands):
        #cache curr island's best fitness
        currIslandBestFitness = islands[i][0]
        
        #if curr island's best fitness is better than best fitness
        if( currIslandBestFitness < bestFitness ):
            #replace best fitness
            bestFitness = currIslandBestFitness
            #copy over curr island's index to save as best island index
            bestFitIslandIndex = i

    #return best fit island
    return islands[bestFitIslandIndex]

#endregion Selection Functs

def Mutate( functionBounds: tuple, child: numpy.ndarray, trait_change_percentage: float ) -> bool:
    """
    Not a guaranteed mutation. 
    Mutation will occur in only 1 in every number of traits of child passed to this function.
    Peforms mutation through increasing/decreasing the child's trait by trait_change_percentage depending on the bounds.  
    Returns true if mutation done.
    Returns false if mutation not done.
    """
    
    #cache useful vals
    num_of_traits = len(child)
    
    #mutate every 1 / number of traits of child
    mutationChance = random.randint(1, num_of_traits )
    if( mutationChance == 1 ):
        #cache bounds
        lower_bound, upper_bound = functionBounds
        
        #change based on bounds
        boundsChange = abs(upper_bound) + abs(lower_bound)
        
        #use a percentage of the change in bounds as the trait change
        stdDev = boundsChange * (trait_change_percentage / 100)
        
        #use a half norm int gaussian distr to choose number of traits to mutate
        xIndexRange, prob = SetupHalfNormIntDistr(num_of_traits, stdDev=1)
        numOfTraitsToMutate = int( numpy.random.choice(xIndexRange, size = 1, p = prob) ) + 1
        
        alreadyMutatedIndices = []
        
        for i in range(0, numOfTraitsToMutate):
        
            #thread safe gauss distr random
            randoGaussChange = random.normalvariate(mu=0, sigma=stdDev)
            
            #decide what child trait being mutated + cache it
            indexBeingMutated = random.randint(0, num_of_traits-1)
            
            #while index being mutated had already been mutated
            while(indexBeingMutated in alreadyMutatedIndices):
                #choose another index to mutate
                indexBeingMutated = random.randint(0, num_of_traits-1)
            
            traitBeingMutated = child[indexBeingMutated]
            
            #apply mutation
            traitBeingMutated += randoGaussChange
            
            #remutate until the trait being mutated is within domain bounds and mutation is nonzero
            while( traitBeingMutated < lower_bound 
                or traitBeingMutated > upper_bound 
                or randoGaussChange == 0
                ):
                
                #reset trait being mutated
                traitBeingMutated = child[indexBeingMutated]
                
                #thread safe gauss distr random
                randoGaussChange = random.normalvariate(mu=0, sigma=stdDev)
                
                #apply mutation
                traitBeingMutated += randoGaussChange
                
            
            """
            #clamp new trait to within bounds
            if( traitBeingMutated < lower_bound):
                traitBeingMutated = lower_bound
            elif( traitBeingMutated > upper_bound ):
                traitBeingMutated = upper_bound
            """
            
            #apply new trait
            child[indexBeingMutated] = traitBeingMutated
            
            alreadyMutatedIndices.append( indexBeingMutated )
        
        #ret true bc mutated
        return True
    
    #return false bc didn't mutate
    return False      

def RunIsland(
    functionEnum: GA_Functions, functionBounds: tuple, pop_size: int, 
    generations: int, num_of_traits: int, parents_elitism_saves: int, 
    parallel_island_model = False, migration_interval = 0, migration_size = 0,
    sender_conn = None, listener_conn = None, results_queue = None,
    show_fitness_plots = False,
    ) -> tuple :
    """Runs an island using the input parameters.

    Returns:
        tuple: bestFitness, bestFitnessData, avgFitnessData, worstFitnessData
    """
    
    #init fitness data space
    worstFitnessData = numpy.empty(generations, dtype=float )
    bestFitnessData = numpy.empty( generations, dtype=float )
    avgFitnessData = numpy.empty( generations, dtype=float )
    #populationFitness = [None] * pop_size
    
    #Sets cannot have two items with the same value.
    solutions = set()
 
    start_time = time.time()
    
    if(parallel_island_model):
        #leave room for migration pop
        local_population_size = pop_size - migration_size
    else:
        local_population_size = pop_size
    
    #create new population
    population = CreatePopulation(
        functionBounds=functionBounds, 
        population_size=local_population_size, #use local pop size incase diff from overall pop size
        individuals_num_of_traits=num_of_traits
    )

    #run for desired generations
    for j in range(0, generations ):

        #popFitnessIndex = migration_size

        localPopFitness = CreateLocalPopulationFitness(functionEnum, population, solutions)
        
        #cache local pop fitness
        #localPopFitness = populationFitness[migration_size:]
            
        #sort in ascending order by fitness (low/good to high/bad)
        localPopFitness.sort(key=getFitness)
        
        #recombo local and migrant pop
        #populationFitness = populationFitness[:migration_size] + localPopFitness
        
        #if doing parallel island model and curr generation is on a migration interval (or first gen)
        if( parallel_island_model and j % (migration_interval-1) == 0 ):
            #take migrant sized section of most fit individuals
            #migrationPopFit = populationFitness[migration_size:migration_size*2]
            
            #choose migrants and store their fitness
            migrationPopFit = ImmigrantSelection(localPopFitness, desiredImmigrants=migration_size)
            
            #pipe it over to the next population
            sender_conn.send(migrationPopFit)
            
            #pipe a keyword to get the listener to stop listening
            
            #listen for more pipe data #until the stop listening keyword is seen
            while True:
                recvdMigrationPopFit = listener_conn.recv()
                if( recvdMigrationPopFit != None):
                    break
            
            #use the recieved migrants to replace the old ones at the front/most fit of the pop

            #recombo local and migrant pop
            populationFitness = recvdMigrationPopFit + localPopFitness #recvdMigrationPopFit.tolist() + localPopFitness.tolist()  #numpy.concatenate( (recvdMigrationPopFit, localPopFitness) )
            #sort resultant mixed pop
            populationFitness.sort(key=getFitness)
        #if not running parrallel island model
        elif( parallel_island_model == False):
            #local pop is our only pop
            populationFitness = localPopFitness

        #print(populationFitness)

        worstFitnessData[j] = max( populationFitness, key=getFitness ).fitness
        bestFitnessData[j] = min( populationFitness, key=getFitness ).fitness

        #find avg (including migrants if any)
        fitnessSum = 0
        for i in range(0, pop_size):
            #take the fitness sum
            fitnessSum += populationFitness[i].fitness            
        avgFitnessData[j] =  fitnessSum/pop_size 
        
        #if not last generation 
        # don't create new pop for last generation bc it won't be eval'd
        if( j < generations - 1):
            
            popIndex = 0
            
            #Create a whole new local pop from prev pop as parents
            for k in range(0, int(local_population_size/2)):
                
                #if less children than parents saved for elitism
                if( k < parents_elitism_saves/2):
                    #apply elitism for next 2 most fit parents
                    children = populationFitness[k].individual, populationFitness[k+1].individual
                
                #not applying elitism
                else:
                    #find parents
                    parents = ParentSelection(populationFitness)

                    #crossover breed parents to get children
                    children = CrossoverBreed(parents[0], parents[1])

                    #walk thru children
                    for child in children:
                        #mutate child 
                        Mutate(
                            functionBounds=functionBounds, child=child,  
                            trait_change_percentage=Implementation_Consts.TRAIT_CHANGE_PERCENTAGE
                        )
                
                #walk thru gen'd children
                for child in children:
                    #add to new population (reuse old space)
                    population[popIndex] = child
                    
                    popIndex += 1
                
            assert popIndex == local_population_size, "Size of population was changed to {}.".format(popIndex)
        
        
    bestFitness = bestFitnessData[generations-1]
    
    #document best fitness per run
    print(
        "Island resulted in a best fitness of " 
        + str(bestFitness) 
        + " for {} in {} seconds.".format( functionEnum, time.time() - start_time)
    )
    
    #if no sols found
    if( len(solutions) == 0 ):
        
        bestFitIndividuals = set()
        
        #store all the best fit individuals in the last generation
        for i in range(0, pop_size):
            if( populationFitness[i].fitness == bestFitness ):
                bestFitIndividuals.add(tuple(populationFitness[i].individual))
    #if sols already found
    else:
        #best fit individuals are the sols
        bestFitIndividuals = solutions
        
    print("Best fit individuals of this run: {}".format(bestFitIndividuals))
    
    if(show_fitness_plots):
            t = numpy.arange(0, generations)
            
            plt.rcParams.update({'font.size': 22})
            plt.plot(t, bestFitnessData) 
            plt.grid() #add a grid to graph
            plt.title('Best Fitness per Generation for {}'.format(functionEnum))
            plt.ylabel('Best Fitness')
            plt.xlabel('Generation')
            plt.show()

            #plt.subplot(3, 1, 2)
            plt.plot(t, avgFitnessData) 
            plt.grid() #add a grid to graph
            plt.title('Average Fitness per Generation for {}'.format(functionEnum))
            plt.ylabel('Average Fitness')
            plt.xlabel('Generation')
            plt.show()

            #plt.subplot(3, 1, 3)
            plt.plot(t, worstFitnessData) 
            plt.grid() #add a grid to graph
            plt.title('Worst Fitness per Generation for {}'.format(functionEnum))
            plt.ylabel('Worst Fitness')
            plt.xlabel('Generation')
            plt.show()

    #if sender conn end of pipe given to funct
    if(sender_conn != None):
        #close sender pipe end
        sender_conn.close()

    #store results in a tuple
    results_tuple = bestFitness, bestFitnessData, avgFitnessData, worstFitnessData

    #if a queue was passed in
    if(results_queue != None):
        #place the results in the queue
        results_queue.put( results_tuple )
        
        #may have to implement a queue lock so sub proc's won't write to queue at same time
        
        #cancel sub proc so won't wait till queue emptied
        #multiprocessing.JoinableQueue.cancel_join_thread()

    #return the results queue
    return results_tuple