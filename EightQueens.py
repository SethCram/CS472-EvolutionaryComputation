from multiprocessing.dummy import Array
from typing import Tuple
import numpy
import matplotlib.pyplot as plt

"""
Instructions:
Collect data on the worst, average, and best fitness within the population at each iteration.
Create visuals of the data and write a short paper detailing your EA and results.

"""

"""
Create a fitness function which will evaluate how 'fit' an individual is by counting up the number of queens attacking each other (lower is more fit). 
"""
def EvalFitness( queen_positions: Array[Tuple()]) -> int:
    return 5

"""
Create a selection function which will select two parents from the population, this should be slightly weighted towards more fit individuals.
"""
def BreedSelection( population: Array[Array[Tuple()]] ) -> Tuple(int, int): #change ret type to Array of ints for scalability?
    return (0, 1)

"""
Create a crossover function, which will accept two individuals (parents), and create two children, which should inherit from the parents.
"""
def CrossoverBreed( parent1: Array[Tuple()], parent2: Array[Tuple()] ) -> Tuple( Array[Tuple()], Array[Tuple()] ): #change input and ret type to Array of Array of Tuples for scalability?
    return Tuple(Array[Tuple(0,1)], Array[Tuple(0,1)])

"""
Create a mutation function, which will have a small probability of changing the values of the new children.
"""
def Mutate( child: Array[Tuple()] ) -> Array[Tuple()]:
    return Array[Tuple(0,1)]

"""
Create a survival function which removes the two worst individuals from the population, and then puts the new children into the population.
"""
def SurvivalReplacement( population: Array[Array[Tuple()]], children: Tuple( Array[Tuple()], Array[Tuple()]) ) -> None: #change children input to array of array tuples for scalability?
    #can replace elements in an array, don't needa return anything
    return
    

POPULATION_SIZE = 100
EVOLVE_ITERATIONS = 1000

queen_positions = numpy.arange()

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