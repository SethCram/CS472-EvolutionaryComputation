import random
from secrets import randbelow
import numpy
from scipy.stats import truncnorm
import scipy.stats as ss
import matplotlib.pyplot as plt
from dataclasses import dataclass

#region Creation Functs

def CreateRandomIndividual(number_of_traits: int, board_size_x: int, board_size_y: int) -> numpy.array:
    """
    Creates an individual with random traits.
    Eight queens prob traits = queens' positions.
    Created queens can't occupy already occupied spaces.
    """
    
    #init arr w/ None
    individual = numpy.array( [None] * number_of_traits )
    
    #walk thru each trait of individual
    for traitIndex in range(0, number_of_traits):
        #find new trait using random coords within bounds
        newTrait = (randbelow(board_size_x), randbelow(board_size_y))
    
        #verify that created position is unique and doesn't occupy another Queen's space
        
        checkIndex = 0
        
        #if not first trait
        if( traitIndex != 0):
            #walk thru all already created traits
            while( checkIndex < traitIndex):
                #if another queen shares the same spot
                if( newTrait[0] == individual[checkIndex][0] and newTrait[1] == individual[checkIndex][1] ):
                    #fill that individual's trait using random x and y coords
                    newTrait = (randbelow(board_size_x), randbelow(board_size_y))
                    #restart duplicate check
                    checkIndex = 0
                #if another queen doesn't share same spot
                else:
                    #move onto next queen
                    checkIndex += 1
        
        #fill that individual's trait
        individual[traitIndex] = newTrait
        
    return individual

def CreatePopulation(population_size: int, number_of_traits: int, board_size_x: int, board_size_y: int) -> numpy.array:
    """
    Create population with random individuals using the given params.
    """
    
    #init pop
    population = numpy.array( [None] * population_size )
    
    #populate every member of population
    for individualIndex in range(0, population_size):
        population[individualIndex] = CreateRandomIndividual(number_of_traits, board_size_x, board_size_y)
    
    return population

#endregion Creation Methods

#region Fitness Functs and Class

"""
Create a fitness function which will evaluate how 'fit' an individual is by counting up the number of queens attacking each other (lower is more fit). 
"""
def EvalFitness( queen_positions: numpy.array(tuple) ) -> int:
    """
    Evaluates fitness of a single individual.
    """
    collisions = 0
    
    numOfQueens = len( queen_positions )
    
    #walk thru every queen on board
    for i in range(0, numOfQueens):
        
        #for every queen on board, compare it to every other queen on the board
        for j in range(0, numOfQueens):
            
            #if not comparing the same queen (queens cant occupy the same space)
            if( i != j ):
                #find the slope tween the two queens
                changeInX, changeInY = (queen_positions[i][0] - queen_positions[j][0], queen_positions[i][1] - queen_positions[j][1])
                
                #make sure 2 queens don't occupy the same spot
                #assert (changeInX == 0 and changeInY == 0) == False
                
                #if queens are on the same x or y axis
                if(changeInX == 0 or changeInY == 0):
                    #incr collisions
                    collisions += 1
                #if queens aren't on same axis
                else:
                    slope = abs( changeInY/changeInX )
                    
                    #if diagonal collision tween Queens bc of slope
                    if( slope == 1):
                        collisions += 1
    
    #ensure num of collisions isn't above the max
    assert collisions <= numOfQueens * numOfQueens
                
    return collisions

@dataclass 
class PopulationFitness:
    """
    Class to keep track of an individual and their fitness score.
    """
    individual: numpy.array(tuple)
    fitness: int

def getFitness( individual: PopulationFitness ) -> int:
        return individual.fitness
    
#endregion Fitness Functs and Class

#region Breeding Functs

"""
Create a selection function which will select two parents from the population, this should be slightly weighted towards more fit individuals.
"""
#def BreedSelection( population: numpy.array[numpy.array[tuple()]] ) -> tuple(int, int): #change ret type to Array of ints for scalability?
def BreedSelection( population: numpy.array, displayDistributionGraph = False ) -> numpy.array(numpy.array(tuple)):
    """
    Assumes population array is sorted in ascending fitness order (low/good to high/bad).
    Returns an array of two parents.
    If displayDistributionGraph is True, random distr shown using random sample data.
    """
    
    #store pop size
    pop_size = len(population)
    
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
    
    """Approach 4: use half norm to take half of the normal funct (every other val is half prob + goes over 100)
    parent1Index = ss.halfnorm( loc=0,scale=30).rvs(size=100000).round().astype(int)
    
    #if overloaded to display distr graph
    if( displayDistributionGraph):
        #display distr graph
        #nums = numpy.random.choice(x, size = 1000000, p = prob)
        plt.hist(parent1Index, bins = pop_size)
        plt.show()
    """
    
    """Approach 3: use straightup truncnorm, conv to int, and graph it (every other val is half prob)
    #use a normal distr to choose 2 parents (inclusive)
    parent1Index = truncnorm(a=0,b=pop_size-1, loc=0,scale=30).rvs(size=100000).round().astype(int)
    parent2Index = truncnorm(a=0,b=pop_size-1, loc=0,scale=30).rvs(size=100000).round().astype(int)
    
    #if overloaded to display distr graph
    if( displayDistributionGraph):
        #display distr graph
        #nums = numpy.random.choice(x, size = 1000000, p = prob)
        plt.hist(parent1Index, bins = pop_size)
        plt.show()
    """
    
    """Approach 2: use truncnorm in place of norm for approach 1. (bad at zero)
    x = numpy.arange(0, pop_size) #bc upper bound is exclusive
    xU, xL = x + 0.5, x - 0.5
    prob = ss.truncnorm.cdf(xU, a=0, b=pop_size-1, scale = 30) - ss.truncnorm.cdf(xL, 0, pop_size-1, scale = 30) #scale represents inner quartiles
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    
    #if overloaded to display distr graph
    if( displayDistributionGraph):
        #display distr graph
        nums = numpy.random.choice(x, size = 1000000, p = prob)
        plt.hist(nums, bins = pop_size)
        plt.show()
    """
    
    """Approach 1: Turn normal distr into half norm using abs value and int conversion (bad at zero)
    #generate a rando normal distr of ints
    x = numpy.arange(-pop_size+1, pop_size) #bc upper bound is exclusive
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = 30) - ss.norm.cdf(xL, scale = 30) #scale represents inner quartiles
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    
    #if overloaded to display distr graph
    if( displayDistributionGraph):
        #display distr graph
        nums = abs(numpy.random.choice(x, size = 1000000, p = prob))
        plt.hist(nums, bins = len(x))
        plt.show()
    
    
    #choose parent indices, make sure they're ints and positive
    parent1Index = int( abs( numpy.random.choice(x, size = 1, p = prob) ) )
    parent2Index = int( abs( numpy.random.choice(x, size = 1, p = prob) ) )
    """
    
    #make sure indices within array range
    assert parent1Index < pop_size and parent2Index < pop_size and type(parent1Index) == int and type(parent2Index) == int
    
    #while parents are the same
    while( parent2Index == parent1Index):
        #refind parent 2
        parent2Index = int( numpy.random.choice(xIndexRange, size = 1, p = prob) )
    
    #get parents using indices
    parent1 = population[parent1Index]
    parent2 = population[parent2Index]
    #store parents in an array
    parentsArray = numpy.array( [None] * 2)
    parentsArray[0] = parent1
    parentsArray[1] = parent2
        
    #return chosen parents
    return parentsArray

"""
Create a crossover function, which will accept two individuals (parents), and create two children, which should inherit from the parents.
"""
#def CrossoverBreed( parent1: numpy.array[tuple()], parent2: numpy.array[tuple()] ) -> tuple( numpy.array[tuple()], numpy.array[tuple()] ): #change input and ret type to Array of Array of tuples for scalability?
def CrossoverBreed( parent1: numpy.array, parent2: numpy.array ) -> numpy.array:
    """
    Assumes both parents have the same number of traits (queens).
    Randomly assign queen positions from each parent, regardless of fitness.
    """
    
    num_of_traits = len(parent1)
    
    #init child arrs
    child1 = numpy.array( [None] *  num_of_traits )
    child2 = numpy.array( [None] *  num_of_traits )
    children = numpy.array( [None] *  2 )
    
    #board_size_x = 8
    #board_size_y = 8
    
    #walk thru traits of parents
    for i in range(0, num_of_traits):
        
        #roll the dice 50/50 on what child gets what parent's trait
        
        #if want waitings based on fitness of certain traits, need to attach a fitness score to each trait/Queen
        
        if( random.randint(0,1) == 1 ):
            #copy matching child to parent trait
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            #copy non-matching child to parent trait
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        """
        checkIndex = 0
        #if not first trait
        if( i != 0):
            #walk thru all traits
            while( checkIndex < i):
                #if another queen shares the same spot
                if( child1[i] == child1[checkIndex] ):
                    #fill that individual's trait using random x and y coords
                    #child1[i] = (randbelow(board_size_x), randbelow(board_size_y))
                    
                    #use the other parent's coord
                    child1[i] = child2[i]
                    #restart duplicate check
                    checkIndex = 0
                #if another queen doesn't share same spot
                else:
                    #move onto next queen
                    checkIndex += 1   
            
            checkIndex = 0
            
            #walk thru all traits
            while( checkIndex < i):
                #if another queen shares the same spot
                if( child2[i] == child2[checkIndex] ):
                    #fill that individual's trait using random x and y coords
                    #child1[i] = (randbelow(board_size_x), randbelow(board_size_y))
                    
                    #use the other parent's coord
                    child2[i] = child1[i]
                    #restart duplicate check
                    checkIndex = 0
                #if another queen doesn't share same spot
                else:
                    #move onto next queen
                    checkIndex += 1  
        """
    children[0] = child1
    children[1] = child2

    return children

#endregion Breeding Functs

"""
Create a mutation function, which will have a small probability of changing the values of the new children.
"""
#def Mutate( child: numpy.array ) -> numpy.array:
def Mutate( child: numpy.array ) -> bool:
    """
    Not a guaranteed mutation. Mutation will occur in only 20% of children passed to this function.
    Returns true if mutation done.
    Returns false if mutation not done.
    """
    randOneToTen = random.randint(1,10)
    #randOneToTwo = random.randint(1,2)
    
    traitBeingMutated = random.randint(0, len(child)-1)
    childTraitBeingMutated = child[traitBeingMutated]
    
    board_size_x = 8
    board_size_y = 8
    number_of_traits = 8
    
    #mutate around 20% of children
    if( randOneToTen <= 2 ):
        """
        #init new coords w/ old vals (one will be overwritten)
        newX = childTraitBeingMutated[0]
        newY = childTraitBeingMutated[1]
        
        
        # 50/50 roll on whether change x or y
        if( randOneToTen == 1):
            # 50/50 roll on whether subtr or add
            if( randOneToTwo == 1):
                #add one
                newX = childTraitBeingMutated[0] + 1
            else:
                #subtr one
                newX = childTraitBeingMutated[0] - 1
                
            #if new x coord is out of bounds
            if( newX < 0 or newX > board_size_x - 1):
                return False
            
            #apply new x
            #child[traitBeingMutated] = ( newX, childTraitBeingMutated[1] )
        else:
            # 50/50 roll on whether subtr or add
            if( randOneToTwo == 1):
                #add one
                newY = childTraitBeingMutated[1] + 1
            else:
                #subtr one
                newY = childTraitBeingMutated[1] - 1
                
            #if new Y coord is out of bounds
            if( newY < 0 or newY > board_size_y - 1 ):
                return False
            
            #apply new y
            #child[traitBeingMutated] = ( childTraitBeingMutated[0], newY )
        """
        
        #get mutated coords
        newX, newY = getMutatedCoords(childTraitBeingMutated)            
        checkIndex = 0
        
        #remutate if new coords not unique on board or out of bounds
            
        #walk thru all traits
        while( checkIndex < number_of_traits):
            #if another queen shares the same spot, or new Y or X coord out of bounds
            if( 
               (checkIndex != traitBeingMutated and newX == child[checkIndex][0] and newY == child[checkIndex][1])
               or (newY < 0 or newY > board_size_y - 1)
               or (newX < 0 or newX > board_size_x - 1) 
               ):
                #remutate and check again for valid mutation
                newX, newY = getMutatedCoords(childTraitBeingMutated)
                checkIndex = 0
            #if another queen doesn't share same spot + new coords in bounds
            else:
                #move onto next queen
                checkIndex += 1
                
        #apply new trait
        child[traitBeingMutated] = (newX, newY)
        #ret true bc mutated
        return True
    
    #return false bc didn't mutate
    return False      

def getMutatedCoords( childTraitBeingMutated: tuple ) -> tuple:
    """
    Get the mutated coordinates based off of the passed in trait +/- one of its x,y positions.
    """
    
    randOneToTwoChangeXY = random.randint(1,2)
    randOneToTwoChangePlusMinus = random.randint(1,2)
    
    oldX = childTraitBeingMutated[0]
    oldY = childTraitBeingMutated[1]
    
    # 50/50 roll on whether change x or y
    if( randOneToTwoChangeXY == 1):
        # 50/50 roll on whether subtr or add
        if( randOneToTwoChangePlusMinus == 1):
            #add one
            newX = oldX + 1
        else:
            #subtr one
            newX = oldX - 1
        
        #return new coords
        return (newX, oldY)
    else:
        # 50/50 roll on whether subtr or add
        if( randOneToTwoChangePlusMinus == 1):
            #add one
            newY = oldY + 1
        else:
            #subtr one
            newY = oldY - 1
            
        #return new coords
        return (oldX, newY)

"""
Create a survival function which removes the two worst individuals from the population, and then puts the new children into the population.
"""
#def SurvivalReplacement( population: numpy.array, children: tuple( numpy.array, numpy.array) ) -> None: #change children input to array of array tuples for scalability?
def SurvivalReplacement( population: numpy.array, children: numpy.array ) -> None:
    """
    Assumes population array is sorted in ascending fitness order (low/good to high/bad).

    Args:
        population (numpy.array): _description_
        children (numpy.array): _description_
    """
    #cache lengths of pop and children
    population_size = len(population)
    numOfChildren = len(children)

    #replace end of pop (best fit/worst off) w/ children
    population[population_size-numOfChildren:population_size] = children #array slicing is exclusive on upper bound
    
    return