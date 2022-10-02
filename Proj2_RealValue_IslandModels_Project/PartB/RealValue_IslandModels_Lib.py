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
    GA + IM + FS
    GA + IM + CR
    GA + FS + CR
    GA + IM + FS + CR

    Write a paper similar to the papers in project 2, skip describing the problems, but go into detail describing your implementation.

    Do not describe your code structures, but describe the algorithm of your implementation. Plot your results.
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
    GENERATIONS_PER_RUN = 200  #100: best fit = 0.583 #1000: best fit = 0.27 #10,000: best fit = 0.448??
    TRAIT_CHANGE_PERCENTAGE = 3
    PAIRS_OF_PARENTS_SAVED_FOR_ELITISM = 1
    NUMBER_OF_ISLANDS = 5
    MIGRATION_INTERVAL = 5
    PAIRS_OF_IMMIGRANTS = 3
    MAX_CHILDREN_KILLED = 10


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

#iteration number is key, value is IM, FS, CR, graph display str settings, line type str
iterationVarConfig ={
    0: (False, False, False, "", '-'),
    1: (True, False, False, ", IM", 'o'),
    2: (False, True, False, ", FS", '+'),
    3: (False, False, True, ", CR", ':'),
    4: (True, True, False, ", IM, FS", '.'),
    5: (True, False, True, ", IM, CR", '--'),
    6: (False, True, True, ", FS, CR", 'v'),
    7: (True, True, True, ", IM, FS, CR", '*')
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

def EvalFitness( functionToOptimize: GA_Functions , individual: numpy.ndarray) -> float:
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

def CrossoverBreed( parent1: numpy.ndarray, parent2: numpy.ndarray ) ->  numpy.ndarray:
    """
    A 1-point crossover.
    Assumes both parents have the same number of traits.
    Returns two children contained in a numpy array.
    """
    
    num_of_traits = len(parent1)
    
    #init child arrs
    #child1 = numpy.empty( num_of_traits, dtype=float ) #need to be below 0 or above 7
    #child2 = numpy.empty( num_of_traits, dtype=float )
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
    Returns the desired number of immigrants using selecting best individual and rest uniformly random.  
    Provided that the passed in populationFitness is already sorted in ascending order.

    Args:
        populationFitness (numpy.ndarray): sorted in ascending order

    Returns:
        list: IndividualFitness pairings
    """
    
    #store pop size
    pop_size = len(populationFitness)
    
    """
    xIndexRange, prob = SetupHalfNormIntDistr(pop_size, stdDev=30)
    
    #randomly select immigrant indices
    immigrantIndices = numpy.random.choice(xIndexRange, size = desiredImmigrants, p = prob)
    
    immigrants = [None] * desiredImmigrants
    
    i = 0
    for immigrantIndex in immigrantIndices:
        #make sure indices within array range
        assert immigrantIndex < pop_size
        
        #copy over into immigrants arr
        immigrants[i] = populationFitness[int(immigrantIndex)]
        i += 1    
    """
    
    immigrantIndices = numpy.random.uniform(low=0, high=pop_size-0.50, size=desiredImmigrants-1 )
    
    immigrants = [None] * desiredImmigrants
    
    immigrants[0] = populationFitness[0]
    i = 1
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

def GetDistTweenIndividuals(individual1: numpy.ndarray, individual2: numpy.ndarray) -> float:

    #make sure both individual's have same number of traits
    assert len(individual1) == len(individual2)
    
    return sum( abs(individual1 - individual2) )

def Survive( child: numpy.ndarray, newGenPopulation: list, desired_pop_size: int, distThreshold: float, desiredNumOfNiches: int) -> bool:
    """Determines whether the provided child survives

    Args:
        child (numpy.ndarray): 
        newGenPopulation (list): All entries need to be filled.
        desired_pop_size (int): used to determine whether child should die.
        distThreshold (float): used to determine how many other individuals within range of child.
        desiredNumOfNiches (int): used to determine whether child should die.

    Returns:
        bool: whether the provided child survives
    """
    #assumes child and population individuals have same number of traits
    
    num_of_traits = len(child)
    
    new_gen_pop_size = len(newGenPopulation)
    
    if( desiredNumOfNiches <= 1):
        return True
    
    #calc individuals allowed within each niche 
    individuals_within_each_niche = int( desired_pop_size / desiredNumOfNiches )
    
    individualsWithinChildThreshold = 0
    
    #walk thru new gen pop
    for individual in newGenPopulation: 
        
        distFromChild = GetDistTweenIndividuals(child, individual)
            
        #if individual's dist from child is within threshold range (exclusive)
        if( distFromChild < distThreshold):
            #add to count
            individualsWithinChildThreshold += 1
            
    #return whether individuals within child threshold is less than num of individuals within each niche
    return individuals_within_each_niche > individualsWithinChildThreshold

    #higher # of individuals within threshold range, higher chance child doesn't survive?
    
def CalcSharedFitness( popFitness: list, individualsIndexInPopFitness: int, sharing_radius: float) -> float:
    """
    Calculated the shared fitness for an individual in relation to others in the population.
    """
    pop_size = len(popFitness)
    
    shSum = 0
    
    individualFitness = popFitness[individualsIndexInPopFitness]
    
    #walk thru pop
    for i in range(pop_size):
        #if not individual evaling SF for
        if( i != individualsIndexInPopFitness ):
            #get dist tween individuals
            distTweenIndividuals = GetDistTweenIndividuals(individualFitness.individual, popFitness[i].individual)
            
            #if dist tween individuals is within sharing radius
            if( distTweenIndividuals < sharing_radius):
                #calc sh as a number tween 0 and 1 exclusive
                shSum += (1 - (distTweenIndividuals/sharing_radius))
                
    #mult since want lower fitness
    return individualFitness.fitness * shSum
                

    
def CreateChildren(populationFitness: numpy.ndarray, functionBounds: tuple) -> numpy.ndarray:
    """Create two children through selecting two parents, crossing them over, and mutating the children.

    Args:
        populationFitness (numpy.ndarray): array of individual-fitness object pairings
        functionBounds (tuple): lower_bound_inclusive, upper_bound_inclusive

    Returns:
        numpy.ndarray: children
    """
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
        
    return children

def RunIsland(
    functionEnum: GA_Functions, functionBounds: tuple, pop_size: int, 
    generations: int, num_of_traits: int, pairs_of_parents_elitism_saves: int, 
    parallel_island_model = False, migration_interval = 0, pairs_of_immigrants = 0,
    sender_conn = None, listener_conn = None, results_queue = None,
    show_fitness_plots = False, crowding: bool = False, fitness_sharing: bool = False,
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
        local_population_size = pop_size - pairs_of_immigrants * 2
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
        
        #if using island model
        if(parallel_island_model):
            
            #if migration interval or 0th gen
            if( j % (migration_interval-1) == 0 ):
                #create new migrant pop
                
                #sort in ascending order by fitness (low/good to high/bad)
                localPopFitness.sort(key=getFitness)
                
                #choose migrants and store their fitness
                migrationPopFit = ImmigrantSelection(localPopFitness, desiredImmigrants=pairs_of_immigrants * 2)
                
                #pipe it over to the next population
                sender_conn.send(migrationPopFit)
                
                #pipe a keyword to get the listener to stop listening
                
                #listen for more pipe data 
                while True:
                    recvdMigrationPopFit = listener_conn.recv()
                    if( recvdMigrationPopFit != None):
                        break
                
                #use the recieved migrants to replace the old ones at the front/most fit of the pop

            #recombo new local and migrant pop
            populationFitness = recvdMigrationPopFit + localPopFitness 
            
            #sort resultant mixed pop
            populationFitness.sort(key=getFitness)
                        
        #if not running parrallel island model
        else:
            #sort in ascending order by fitness (low/good to high/bad)
            localPopFitness.sort(key=getFitness)
            
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
            
            population = []
            
            #Save parents for elitism 
            for k in range(0, pairs_of_parents_elitism_saves):
                #apply elitism for next 2 most fit parents
                children = populationFitness[k].individual, populationFitness[k+1].individual
            
                for child in children:
                    #add to new population 
                    population.append( child )
                    
                    popIndex += 1
            
            #if using FS
            if(fitness_sharing):
                #walk thru pop fitness
                for i in range(local_population_size):
                    
                    #redef fitness
                    populationFitness[i].fitness = CalcSharedFitness(populationFitness, i, sharing_radius=1)
                    
                #sort in ascending order by fitness (low/good to high/bad)
                populationFitness.sort(key=getFitness)
                
            pairs_of_children = int(local_population_size/2)
            
            #Create rest of whole new local pop from prev pop as parents
            for j in range(pairs_of_parents_elitism_saves, pairs_of_children):    

                #create two new children
                children = CreateChildren(populationFitness, functionBounds)
                
                #if using crowding
                if(crowding):
                    #walk thru newly created children
                    for i in range(len(children)):
                        
                        childrenKilled = 0
                    
                        #while child doesn't survive and not enough children killed
                        while( 
                            Survive(children[i], population, desired_pop_size=Implementation_Consts.POPULATION_SIZE, distThreshold=0.1, desiredNumOfNiches=50) == False
                            and childrenKilled <= Implementation_Consts.MAX_CHILDREN_KILLED
                        ):
                            childrenKilled += 1
                            
                            #create more childen but only use the first one
                            children[i] = CreateChildren(populationFitness, functionBounds)[0]
                
                for child in children:
                    #add to new population 
                    population.append( child )
                    
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

    #return the results queue
    return results_tuple