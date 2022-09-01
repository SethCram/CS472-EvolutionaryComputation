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
    Created queens can occupy already occupied spaces.
    """
    
    #init arr w/ None
    individual = numpy.array( [None] * number_of_traits )
    
    #walk thru each trait of individual
    for traitIndex in range(0, number_of_traits):
        #fill that individual's trait using random x and y coords
        individual[traitIndex] = (randbelow(board_size_x), randbelow(board_size_y))
        
        """
        #need to verify that created position is unique and doesn't occupy another Queen's space
        
        #if not first trait
        if( traitIndex != 0):
            #walk thru all already created traits
            for checkIndex in range(0, traitIndex):
                #if another queen shared the same spot
                if( individual[traitIndex][0] == individual[checkIndex][0] and individual[traitIndex][1] == individual[checkIndex][1] ):
                    #fill that individual's trait using random x and y coords
                    individual[traitIndex] = (randbelow(board_size_x), randbelow(board_size_y))
        """
        
    return individual

def CreatePopulation(population_size: int, number_of_traits: int, board_size_x: int, board_size_y: int) -> None:
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
def EvalFitness( queen_positions: numpy.array) -> int:
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
                
                #if 2 queens occupying same spot:
                if(changeInX == 0 or changeInY == 0):
                    collisions += 1
                #if 2 queens not on same spot
                else:
                    slope = abs( changeInY/changeInX )
                    
                    #if horizontal, vertical, or diagonal collision tween Queens bc of slope
                    if( slope in [0, 0.5, 1]):
                        collisions += 1
    
    #ensure num of collisions isn't above the max
    assert collisions <= numOfQueens * numOfQueens
                
    return collisions

@dataclass 
class PopulationFitness:
    """
    Class to keep track of all individuals and their fitness.
    """
    individual: numpy.array
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
    
    #use a normal distr to choose 2 parents (inclusive)
    #parent1Index = truncnorm(a=0,b=pop_size-1, loc=0,scale=1).rvs(size=1000).round().astype(int)
    #parent2Index = truncnorm(a=0,b=pop_size-1, loc=0,scale=1).rvs(size=1000).round().astype(int)
    
    #generate a rando normal distr of ints
    x = numpy.arange(-pop_size, pop_size+1)
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
    
    #make sure indices within array range
    assert parent1Index < pop_size and parent2Index < pop_size and type(parent1Index) == int and type(parent2Index) == int
    
    #if parents are the same
    while( parent2Index == parent1Index):
        #refind parent 2
        parent2Index = int( abs( numpy.random.choice(x, size = 1, p = prob) ) )
        
    population[0]
    
    parent1 = population[parent1Index]
    parent2 = population[parent2Index]
    
    parentsArray = numpy.array( [None] * 2)
    parentsArray[0] = parent1
    parentsArray[1] = parent2
        
    #return chosen parents
    return parentsArray
    #return numpy.array[parent1Index, parent2Index]

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
            
    children[0] = child1
    children[1] = child2

    return children

#endregion Breeding Functs

"""
Create a mutation function, which will have a small probability of changing the values of the new children.
"""
#def Mutate( child: numpy.array ) -> numpy.array:
def PossiblyMutate( child: numpy.array ) -> bool:
    """
    Not a guaranteed mutation. Mutation will occur in only 20% of children passed to this function.
    Returns true if mutation done succeeded or no mutation.
    Returns false if mutation supposed to be applied but failed.
    """
    randOneToTen = random.randint(1,10)
    randOneToTwo = random.randint(1,2)
    
    traitBeingMutated = random.randint(0, len(child)-1)
    childTraitBeingMutated = child[traitBeingMutated]
    
    board_size_x = 8
    board_size_y = 8
    
    #mutate around 20% of children
    if( randOneToTen <= 2 ):
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
            child[traitBeingMutated] = ( newX, childTraitBeingMutated[1] )
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
            child[traitBeingMutated] = ( childTraitBeingMutated[0], newY )
    
    return True
            
            
        
    #numpy.array[tuple(0,1)]
    return

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
    population[population_size-numOfChildren-1:population_size-1] = children
    
    return

#queen_positions = numpy.arange()

"""EXAMPLES
#plot1:
plt.figure(figsize = (10,15)) #1st arg = horizontal space, 2nd arg = vertical space
plt.subplot(3, 1, 1)
plt.plot(t, f1) # t = step size, smaller it is the more accurate graph is
                # y = array of cos vals at each step
plt.grid() #add a grid to graph
plt.title('f1(t) vs t')
plt.ylabel('f1(t)')
"""

"""
steps = 1e-2 # Define step size
t = numpy.arange(0, 20 + steps , steps)

data = (3, 6, 9, 12)

fig, simple_chart = plt.subplots()

simple_chart.plot(data)

plt.show()
"""