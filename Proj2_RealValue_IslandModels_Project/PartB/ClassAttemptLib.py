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
    
    PARENTS_SAVED_FOR_ELITISM = 2
    assert PARENTS_SAVED_FOR_ELITISM % 2 == 0, "Need to save an even number of parents for elitism."
    assert PARENTS_SAVED_FOR_ELITISM < POPULATION_SIZE, "Can't save more parents for elitism than individuals in the population."

    NUMBER_OF_ISLANDS = 5
    MIGRATION_INTERVAL = 5
    MIGRATION_SIZE = 6
    assert MIGRATION_SIZE % 2 == 0, "Need to save an even number of migrants for new generation."
    assert MIGRATION_SIZE < POPULATION_SIZE, "Can't select more migrants than individuals in the population."
    
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

#endregion GA enum and dicts
class Individual():
    def __init__(self, num_of_traits: int, mutationStdDev: float):
        self.num_of_traits = num_of_traits
        self.fitness = self.EvalFitness()
    def EvalFitness(self):
        pass
    def Mutate(self):
        pass
    
    def __str__(self):
        return f"{self.name}({self.age})"      
      
class Population():
    def __init__(self, benchmark_funct: GA_Functions, popSize: int, mutationStdDev: float):
        self.PopulationSize = popSize
    
    def crossover(self, parent1: Individual, parent2: Individual):
        pass
    def findMigrants(self, migrationSize):
        pass
    def addMigrants(self):
        pass
    def survive(self):
        pass
    
    def __str__(self):
        return f"{self.name}({self.age})"
    

    
class GA():
    def __init__(self, benchmark_funct: GA_Functions, popSize: int, mutationStdDev: float):
        self.PopulationSize = popSize
        self.MutationStdDev = mutationStdDev
        #self.crossoverType
        #self.selectionType
        self.CurrentGeneration = 0
        self.population = []
        self.fitness = []
        
    def runGen(self) -> Population:
        pass
    
    def __str__(self):
        return f"{self.benchmark_funct} with pop size ({self.popSize})"