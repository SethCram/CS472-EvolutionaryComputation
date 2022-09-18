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

#from enum import Enum
from enum import Enum
import random
from secrets import randbelow
import numpy
import scipy.stats as ss
import matplotlib.pyplot as plt
from dataclasses import dataclass
import bisect

""""
class Function_Bounds_Pairings():
    Spherical = (-5.12, 5.12)
    Rosenbrock = (-2.048, 2.048)
    Rastrigin = (-5.12, 5.12)
    Schwefel2 = (-512.03, 511.97)
    Ackley = (-30, 30)
    Griewangk = (-600, 600)
"""
    
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
    
    """
    #dictionary of funct-domain bounds pairings
    functionBoundsDict = { 
        GA_Function.Spherical: (-5.12, 5.12),
        GA_Function.Rosenbrock: (-2.048, 2.048),
        GA_Function.Rastrigin: (-5.12, 5.12),
        GA_Function.Schwefel2: (-512.03, 511.97),
        GA_Function.Ackley: (-30, 30),
        GA_Function.Griewangk: (-600, 600)
    }
    #use dict to determin lower + upper bounds
    lower_bound, upper_bound = functionBoundsDict[functionToOptimize]
    """
    
    lower_bound, upper_bound = functionBounds
    
    #populate every member of population
    for individualIndex in range(0, population_size):
        population[individualIndex] = CreateRandomIndividual( individuals_num_of_traits, lower_bound, upper_bound)
    
    return population

#endregion Creation Methods

#region Fitness Functs and Class

def SqrdSum( inputArr: numpy.ndarray ) -> float:
    """
    Returns the elly-wise squared sum of the arr.
    """
    return sum(inputArr**2)

def CosOfTwoPiTrait( trait: float) -> float:
    return numpy.cos(2*numpy.pi*trait)

def EvalFitness( functionToOptimize: GA_Functions , individual: numpy.ndarray ) -> int:
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
            rslt += ( 100 * ( (rslt[i+1] - currTrait**2)**2 ) + (currTrait-1)**2 )
    
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
            rslt += currTrait*numpy.sin(numpy.sqrt(abs(currTrait)))
        
        #make sure fitness is positive
        rslt = abs(rslt)
        
    elif( functionToOptimize == GA_Functions.Ackley):
        #init w/ added val outside of summation
        rslt = 20 + numpy.e
        
        cosOfTwoPiTraitSum = 0
        
        for i in range(0,num_of_traits):
            #cache curr trait
            currTrait = individual[i]
            #take summation of a custom funct
            cosOfTwoPiTraitSum += CosOfTwoPiTrait(currTrait)
            
        rslt -= ( 
            20 * numpy.exp( -0.2*numpy.sqrt((1/num_of_traits)*SqrdSum(individual)) ) 
            - numpy.exp( (i/num_of_traits)*cosOfTwoPiTraitSum ) 
        )
        
        #make sure fitness is positive
        rslt = abs(rslt)
        
    elif( functionToOptimize == GA_Functions.Griewangk):
        rslt = 1
        
        rslt += sum( (individual**2) / 4000 )
        
        prodOfCosines = 1
        
        #go 1 to num_of_traits inclusive
        for i in range(1,num_of_traits+1):
            #cache curr trait
            currTrait = individual[i]
            
            #take the product over the individual's traits
            prodOfCosines *= numpy.cos(currTrait/numpy.sqrt(i))
        
        rslt -= prodOfCosines    
        
        #make sure fitness is positive
        rslt = abs(rslt)
    
    #desired val is 0 so fitness should always be positive             
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

#region Breeding Functs

"""
Create a selection function which will select two parents from the population, this should be slightly weighted towards more fit individuals.
"""
def BreedSelection( populationFitness: list, displayDistributionGraph = False ) -> numpy.ndarray:
    """
    Assumes population array is sorted in ascending fitness order (low/good to high/bad).
    Returns an array of two parents.
    If displayDistributionGraph is True, random distr shown using random sample data.
    """
    
    #store pop size
    pop_size = len(populationFitness)
    
    """
    Approach 5: tried using half norm and incr'd to take interval 1-100 then subtr 1 after.
    """
    
    #take interval 1-100
    x = numpy.arange(1, pop_size+1) #bc upper bound is exclusive
    #store every number's +/-0.5
    xU, xL = x + 0.5, x - 0.5 
    #determine probability
    prob = ss.halfnorm.cdf(xU, scale = 30) - ss.halfnorm.cdf(xL, scale = 30) #scale represents inner quartiles
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    #decr by 1 to find the index 0-99
    xIndexRange = x - 1
    
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
    parentsArray[0] = parent1
    parentsArray[1] = parent2
        
    #return chosen parents
    return parentsArray

"""
Create a crossover function, which will accept two individuals (parents), and create two children, which should inherit from the parents.
"""
def CrossoverBreed( parent1: numpy.ndarray, parent2: numpy.ndarray ) -> numpy.ndarray:
    """
    Assumes both parents have the same number of traits (queens).
    A variation on 1-point crossover.
    """
    
    num_of_traits = len(parent1)
    
    outsideRangeDefault = 20
    
    #init child arrs
    child1 = numpy.full( num_of_traits, outsideRangeDefault, dtype=int ) #need to be below 0 or above 7
    child2 = numpy.full( num_of_traits, outsideRangeDefault, dtype=int )
    children = numpy.array( [None] * 2 )
    
    #crossover point
    xpoint = random.randrange(1,7) #don't want at very beginning or end bc don't wanna copy parents
    
    #copy over start of each parent
    child1[:xpoint] = parent1[:xpoint]
    child2[:xpoint] = parent2[:xpoint]
    
    #get tails for each child from other parent
    OnePointTailCrossover( child=child1, xpoint=xpoint, parent=parent2, outsideRangeDefault=outsideRangeDefault )
    OnePointTailCrossover( child=child2, xpoint=xpoint, parent=parent1, outsideRangeDefault=outsideRangeDefault)

    #place childs into children arr
    children[0] = child1
    children[1] = child2

    return children

def OnePointTailCrossover( parent: numpy.ndarray, xpoint: int, child: numpy.ndarray, outsideRangeDefault: int) -> None:
    """A 1-point crossover performed with the given parent and child at the xpoint. 
        No individual can have more than one of the same trait.

    Args:
        parent (numpy.ndarray): Parent the child's traits are taken from.
        xpoint (int): Crossover point the tail starts on (inclusive).
        child (numpy.ndarray): Child needing its tail filled.
        outsideRangeDefault (int): Value for error checking to make sure child filled.
    """
    
    num_of_traits = len(parent)
    
    parentIndexIncrs = 0

    #fill in each child's tail w/ the other parent
    for tailIndex in range(xpoint, num_of_traits):
        
        #if start of for loop/child filling
        if(tailIndex == xpoint):
            #start parent index at tail index
            parentIndex = tailIndex
        
        #wait till full loop around
        while( parentIndexIncrs < num_of_traits ):
        
            parentCandidateTrait = parent[parentIndex] 
            
            #look for nxt parent index
            parentIndex += 1
            parentIndexIncrs += 1
            
            #if parent index reached the end
            if(parentIndex == num_of_traits):
                #wrap it around
                parentIndex = 0
            
            #if trait not in child already
            if( parentCandidateTrait not in child ):
                #copy it over
                child[tailIndex] = parentCandidateTrait
                #move onto nxt child trait that needs filling
                break
    
    #make sure all vals in child replaced
    assert outsideRangeDefault not in child, "Not all traits in child replaced by a parent trait: {}".format(child)
    
    return

#endregion Breeding Functs

"""
Create a mutation function, which will have a small probability of changing the values of the new children.
"""
def Mutate( child: numpy.ndarray ) -> bool:
    """
    Not a guaranteed mutation. 
    Mutation will occur in only 1 in every (2*number of traits of child) passed to this function.
    Peforms mutation through swapping 2 random traits of passed in child.  
    Returns true if mutation done.
    Returns false if mutation not done.
    """
    num_of_traits = len(child)
    
    #mutate 1 in every (2*number of traits of child)
    mutationChance = random.randint(1, num_of_traits * 2 )
    if( mutationChance == 1 ):
        
        traitBeingSwapped= random.randint(0, num_of_traits-1)
        otherTraitBeingSwapped= random.randint(0, num_of_traits-1)
        childTraitBeingMutated = child[traitBeingSwapped]
        
        #apply new traits thru swapping
        child[traitBeingSwapped] = child[otherTraitBeingSwapped]
        child[otherTraitBeingSwapped] = childTraitBeingMutated
        
        #ret true bc mutated
        return True
    
    #return false bc didn't mutate
    return False      

"""
Create a survival function which removes the two worst individuals from the population, and then puts the new children into the population.
"""
def SurvivalReplacement( populationFitness: list, children: numpy.ndarray ) -> None:
    """
    Evaluates the newly created childrens' fitness, then uses an insertion sort to add them to the list.
    Assumes population array is sorted in ascending fitness order (low/good to high/bad).

    Args:
        population (list): list of individual fitness objects
        children (numpy.ndarray): array of two children individual fitness objects
    """
    
    for child in children:
        childFitness = IndividualFitness(child, EvalFitness(child))
        
        #insertion sort to pop data and pop
        bisect.insort(populationFitness, childFitness, key=getFitness)
    
    return