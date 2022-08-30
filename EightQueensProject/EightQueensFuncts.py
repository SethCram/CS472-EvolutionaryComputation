#from multiprocessing.dummy import Array
from secrets import randbelow
#from typing import Tuple
import numpy

def CreateRandomIndividual(number_of_traits: int, board_size_x: int, board_size_y: int) -> numpy.array:
    """
    Creates an individual with random traits.
    eight queens prob traits = queens' positions
    """
    
    #init arr w/ None
    individual = numpy.array( [None] * number_of_traits )
    
    #walk thru each trait of individual
    for traitIndex in range(0, number_of_traits):
        #fill that individual's trait using random x and y coords
        individual[traitIndex] = (randbelow(board_size_x), randbelow(board_size_y))
    
    return individual

def CreatePopulation(population_size: int, number_of_traits: int, board_size_x: int, board_size_y: int) -> None:
    """
    Create population of the desired size with random individuals.
    """
    
    #init pop
    population = numpy.array( [None] * population_size )
    
    #populate every member of population
    for individualIndex in range(0, population_size):
        population[individualIndex] = CreateRandomIndividual(number_of_traits, board_size_x, board_size_y)
        
        #need to verify that created individual is unique and doesn't occupy another Queen's space
    
    return population

"""
Create a fitness function which will evaluate how 'fit' an individual is by counting up the number of queens attacking each other (lower is more fit). 
"""
def EvalFitness( queen_positions: numpy.array) -> int:
    
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
                slope = abs( changeInY/changeInX )
                
                #if horizontal, vertical, or diagonal collision tween Queens bc of slope
                if( slope in [0, 0.5, 1]):
                    collisions += collisions
                
    return collisions

"""
Create a selection function which will select two parents from the population, this should be slightly weighted towards more fit individuals.
"""
#def BreedSelection( population: numpy.array[numpy.array[tuple()]] ) -> tuple(int, int): #change ret type to Array of ints for scalability?
def BreedSelection( population: numpy.array ) -> numpy.array:
    return numpy.array[0, 1]

"""
Create a crossover function, which will accept two individuals (parents), and create two children, which should inherit from the parents.
"""
#def CrossoverBreed( parent1: numpy.array[tuple()], parent2: numpy.array[tuple()] ) -> tuple( numpy.array[tuple()], numpy.array[tuple()] ): #change input and ret type to Array of Array of tuples for scalability?
def CrossoverBreed( parent1: numpy.array, parent2: numpy.array ) -> numpy.array:
    return numpy.array[numpy.array[tuple(0,1)], numpy.array[tuple(0,1)]]

"""
Create a mutation function, which will have a small probability of changing the values of the new children.
"""
#def Mutate( child: numpy.array ) -> numpy.array:
def Mutate( child: numpy.array ) -> None:
    #numpy.array[tuple(0,1)]
    return

"""
Create a survival function which removes the two worst individuals from the population, and then puts the new children into the population.
"""
#def SurvivalReplacement( population: numpy.array, children: tuple( numpy.array, numpy.array) ) -> None: #change children input to array of array tuples for scalability?
def SurvivalReplacement( population: numpy.array, children: numpy.array ) -> None:
    #can replace elements in an array, don't needa return anything
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