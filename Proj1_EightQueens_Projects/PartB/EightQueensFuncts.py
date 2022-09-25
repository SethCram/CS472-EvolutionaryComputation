"""
Author: Seth Cram
Class: Evolutionary Computation - CS472/CS572
Project 1 part b - 8 Queen puzzle
Instructions:
    Collect data on the worst, average, and best fitness within the population at each iteration.
    Create visuals of the data and write a short paper detailing your EA and results.
"""
import random
from secrets import randbelow
import numpy
import scipy.stats as ss
import matplotlib.pyplot as plt
from dataclasses import dataclass
import bisect

#region Creation Functs

def CreateRandomIndividual( board_size_x: int, board_size_y: int) -> numpy.ndarray:
    """
    Creates an individual with random traits.
    Eight queens prob traits = queens' positions.
    Created queens can't occupy already occupied spaces.
    """
    
    #init arr w/ enough space
    individual = numpy.empty( board_size_x, dtype=int )
    
    #walk thru each trait of individual
    for traitIndex in range(0, board_size_x):
        #find new trait using random coords within bounds
        newTrait = randbelow(board_size_y)
    
        #verify that created position is unique and doesn't occupy another Queen's space
        
        checkIndex = 0
        
        #if not first trait
        if( traitIndex != 0):
            #walk thru all already created traits
            while( checkIndex < traitIndex):
                #if another queen shares the same spot
                if( newTrait == individual[checkIndex] ):
                    #fill that individual's trait using random x and y coords
                    newTrait = randbelow(board_size_y)
                    #restart duplicate check
                    checkIndex = 0
                #if another queen doesn't share same spot
                else:
                    #move onto next queen
                    checkIndex += 1
        
        #fill that individual's trait
        individual[traitIndex] = newTrait
        
    return individual

def CreatePopulation(population_size: int, board_size_x: int, board_size_y: int) -> numpy.ndarray:
    """
    Create population with random individuals using the given params.
    """
    
    #init pop
    population = numpy.array( [None] * population_size )
    
    #populate every member of population
    for individualIndex in range(0, population_size):
        population[individualIndex] = CreateRandomIndividual( board_size_x, board_size_y)
    
    return population

#endregion Creation Methods

#region Fitness Functs and Class

"""
Create a fitness function which will evaluate how 'fit' an individual is by counting up the number of queens attacking each other (lower is more fit). 
"""
def EvalFitness( queen_positions: numpy.ndarray ) -> int:
    """
    Evaluates fitness of a single individual.
    Queens can't occupy the same space.
    """
    collisions = 0
    
    numOfQueens = len( queen_positions )
    
    #walk thru every queen on board
    for i in range(0, numOfQueens):
        
        #for each queen, use some stats
        checkingQueen = queen_positions[i]
        diag1 = False
        diag2 = False
        diag3 = False
        diag4 = False
        
        #for every queen on board, compare it to every other queen on the board
        for j in range(0, numOfQueens):
            
            #if not comparing the same queen 
            if( i != j ):
                               
                otherQueen = queen_positions[j]
                
                #find the x,y diffs tween the two queens
                # change in x is index difference bc the col pos is index
                changeInX, changeInY = (j - i, otherQueen - checkingQueen)
                
                #store state of the x,y queen changes
                posChangeInX = changeInX > 0
                negChangeInX = changeInX < 0
                posChangeInY = changeInY > 0
                negChangeInY = changeInY < 0
                
                #calc positive slope
                slope = abs( changeInY/changeInX )
                
                #if diagonal collision
                if( slope == 1 ):
                    #if 1st quadrant diagonal collision
                    if(posChangeInX and posChangeInY and diag1 == False):
                        collisions += 1
                        diag1 = True
                    #if 2nd quadrant diagonal collision
                    elif( negChangeInX and posChangeInY and diag2 == False ):
                        collisions += 1
                        diag2 = True
                    #if 3rd quadrant diagonal collision
                    elif( negChangeInX and negChangeInY and diag3 == False ):
                        collisions += 1
                        diag3 = True
                    #if 4th quadrant diagonal collision
                    elif( posChangeInX and negChangeInY and diag4 == False ):
                        collisions += 1
                        diag4 = True
                
                #make sure 2 queens aren't horizontal and vertical of one another
                assert (changeInX != 0 and changeInY != 0) == True, "Queen {} and {} are vertical or horizontal of one another.".format(j, i)
    
    #ensure num of collisions for this board isn't above the max
    assert collisions <= (numOfQueens - 1) * 4, "More than the maximum number of collisions occurred."
                
    return collisions

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
def ParentSelection( populationFitness: list, displayDistributionGraph = False ) -> numpy.ndarray:
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
    Evaluates the newly created childrens' fitness, then uses an insertion sort to add them to the list, and cut out the last two elements.
    Assumes population array is sorted in ascending fitness order (low/good to high/bad).

    Args:
        population (list): list of individual fitness objects
        children (numpy.ndarray): array of two children individual fitness objects
    """
    #cache lengths of pop and children
    population_size = len(populationFitness)
    
    for child in children:
        childFitness = IndividualFitness(child, EvalFitness(child))
        
        #insertion sort to pop data and pop
        bisect.insort(populationFitness, childFitness, key=getFitness)
    
    #cut out two worst fitness individuals
    populationFitness[:] = populationFitness[0:population_size]
    
    return