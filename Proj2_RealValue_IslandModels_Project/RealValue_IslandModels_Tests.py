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
import unittest

from RealValue_IslandModels_Lib import *

class TestCreationMethods(unittest.TestCase):

    def test_IndividualCreation(self):
        """
        Tests individual creation function once for every function.
        Ensures each generated trait is a 64 bit numpy float, 
        within the function's domain,
        and has the correct number of traits.
        """
        
        #for function in GA_Functions:
        
        #walk thru funct bounds dict
        for functionEnum, functionBounds in functionBoundsDict.items():
            
            #store bounds
            lower_bound, upper_bound = functionBounds
            #create rando individual
            
            testIndividual = CreateRandomIndividual(
                num_of_traits=INDIVIDUALS_NUMBER_OF_TRAITS, 
                lower_bound_inclusive=lower_bound,
                upper_bound_inclusive=upper_bound
            )
            
            #trait number check
            self.assertTrue(
                len(testIndividual) == INDIVIDUALS_NUMBER_OF_TRAITS,
                "A test trait for {} doesn't have {} traits.".format(functionEnum, INDIVIDUALS_NUMBER_OF_TRAITS) 
            )
            
            #walk thru individual's traits
            for testTrait in testIndividual:
                #bounds check
                self.assertTrue( 
                    testTrait >= lower_bound and testTrait <= upper_bound, 
                    "A test trait for {} is out of bounds.".format(functionEnum) 
                )
                #type check
                testTraitType = type(testTrait)
                self.assertTrue(
                    testTraitType == numpy.float64,
                    "A test trait for {} isn't a float. It's a {}.".format(functionEnum, testTraitType) 
                )

    def test_PopulationCreation(self):
        """
        Creates a population for each funct 
        and verifies it's the right size.
        """
        #walk thru funct bounds dict
        for functionEnum, functionBounds in functionBoundsDict.items():
            #create new population
            population = CreatePopulation(
                functionBounds=functionBounds, 
                population_size=POPULATION_SIZE, 
                individuals_num_of_traits=INDIVIDUALS_NUMBER_OF_TRAITS
            )
            
            newPopSize = len(population)
            
            #pop size check
            self.assertTrue(
                newPopSize == POPULATION_SIZE,
                "Size of population was changed to {} for {}.".format(newPopSize, functionEnum)
            )

class TestFitnessRelated(unittest.TestCase):
    def test_EvalFitness(self):
        """
        Test to make sure an individual with the target input as traits 
        results in a fitness of zero for every function.
        """
        
        #walk thru funct bounds dict
        for functionEnum, functionTargetInput in functionInputTargetDict.items():
            
            #testIndividual = numpy.full(INDIVIDUALS_NUMBER_OF_TRAITS, functionTargetInput, dtype=float)
            testIndividual = numpy.full(INDIVIDUALS_NUMBER_OF_TRAITS, functionTargetInput, dtype=numpy.float64)
            
            testIndividualFitness = EvalFitness(functionEnum, testIndividual)
            
            errorMsg = "When evaluated with an individual of all {}, {} resulted in a fitness of {}, not zero.".format(functionTargetInput, functionEnum, testIndividualFitness)
            
            self.assertEqual( 
                testIndividualFitness,
                0,
                errorMsg
            )
            
class TestSelectionRelatedMethods(unittest.TestCase):
    def test_SetupHalfNormIntDistr(self):
        """
        Tests the half normal integer distribution parent indices are drawn from.
        Verifies that a lower index number occures more often than a higher index number.
        """
        
        testDataCount = 1000000
        
        xIndexRange, prob = SetupHalfNormIntDistr(POPULATION_SIZE)
        
        #take rando nums using the calc'd prob and index range
        randoNums = numpy.random.choice(xIndexRange, size = testDataCount, p = prob)

        #get count of each unique num gen'd
        unique, counts = numpy.unique(randoNums, return_counts=True)
        
        #pair count of each index in a dict
        indexCountDict = dict(zip(unique, counts))
        
        compIndex1 = 0
        compIndex2 = int(POPULATION_SIZE/3)
        
        #ensure index 0 occured more often than the comp2 index
        self.assertGreater(
            indexCountDict[compIndex1],
            indexCountDict[compIndex2],
            "A lower index in the distribution should usually occure more often."
        )

    def test_BreedSelection(self):
        """
        Tests that all parents generated by the selection function 
        come from the population. 
        #If told to display the distribution graph the selection function displays a plot.
        """
        
        populationFitness = [None] * POPULATION_SIZE
        
        for functionEnum, functionBounds in functionBoundsDict.items():
        
            #create new population
            population = CreatePopulation(
                functionBounds=functionBounds, 
                population_size=POPULATION_SIZE, 
                individuals_num_of_traits=INDIVIDUALS_NUMBER_OF_TRAITS
            )

            #walk thru each individual in pop
            for i in range(0, POPULATION_SIZE):
                individual = population[i]
                individualFitness = EvalFitness(functionEnum, individual)
                
                #store individual w/ their fitness data
                populationFitness[i] = IndividualFitness( individual, individualFitness )
            
            #get parents from pop-fitness obj    
            parents = BreedSelection(populationFitness)
            
            #walk thru parents
            for parent in parents:
                #make sure each parent is actually in the pop
                self.assertTrue(
                    any((numpy.array_equal(parent, individual)) for individual in population),
                    "One of the selected parents isn't in the population."
                )
            
            """
            #cache figures gen'd by matplotlib
            #prevPlottedFigures = plt.get_fignums()
            prevPlottedFigures = plt.gcf().number
            
            #show distr figure for parents
            BreedSelection(populationFitness, displayDistributionGraph=True)

            #cache figures gen'd by matplotlib
            #afterPlottedFigures = plt.get_fignums()
            afterPlottedFigures = plt.gcf().number
            
            #close any open plots (only works sometimes??)
            plt.close()
            
            #make sure a plot was actually displayed
            self.assertGreater( afterPlottedFigures, prevPlottedFigures,
                "Plot never generated to show Breed Selection parent index distribution."    
            )
            """
            
class TestCrossoverRelated(unittest.TestCase):
    
    def test_NPointCrossover(self):
        """
        Tests n-point crossover through crossing over 2 parents 
        and making sure all their children have traits from one of them
        and neither of them are clones of one of their parents.
        """
        
        #should repeatedly do this a number of times for verification
    
        parent1Val = 0
        parent2Val = 1
    
        #init parent array w/ 0's
        parent1 = numpy.full(INDIVIDUALS_NUMBER_OF_TRAITS, parent1Val)
        
        #init parent array w/ 1's
        parent2 = numpy.full(INDIVIDUALS_NUMBER_OF_TRAITS, parent2Val)
        
        #create new children using parents
        children = CrossoverBreed(parent1, parent2)
        
        #walk thru new children
        for child in children:
            """
            #walk thru every child's traits
            for trait in child:
                #ensure every child's trait is one of the parents
                self.assertTrue(
                    trait == parent1Val or trait == parent2Val,
                    "A child contains a trait not from either of their parents."
                )
            """
            
            #get count of each unique num gen'd
            unique, counts = numpy.unique(child, return_counts=True)
            
            #pair count of each index in a dict
            traitCountDict = dict(zip(unique, counts))
            
            #cache parent vals
            parent1Vals = traitCountDict[parent1Val]
            parent2Vals = traitCountDict[parent2Val]
            
            #ensure child isn't clone of either parent
            self.assertTrue(
                parent1Vals < INDIVIDUALS_NUMBER_OF_TRAITS
                and parent2Vals < INDIVIDUALS_NUMBER_OF_TRAITS,
                "This child is a clone of one of its parents."
            )
            
            #ensure all child traits from one of its parents
            self.assertTrue(
                parent1Vals + parent2Vals == INDIVIDUALS_NUMBER_OF_TRAITS,
                "One of the child's traits aren't from either parent."
            )

class TestMutateRelated(unittest.TestCase):
    
    def test_Mutate(self):
        """
        Tests mutate function through repeatedly calling it 
        until a mutation occures.
        Ensures a mutation happened and it didn't 
        cause any of the child's traits to go out of bounds.
        """
        
        #should run multiple times for bounds verification
        
        individualToMutate = numpy.ones(INDIVIDUALS_NUMBER_OF_TRAITS)
        
        unMutatedIndividual = individualToMutate
        
        lower_bound = -1
        upper_bound = 1
        
        #loop till successful mutation
        while(Mutate((lower_bound, upper_bound), individualToMutate) == False):
            pass
    
        #ensure individual mutated
        self.assertTrue( 
            numpy.array_equal(unMutatedIndividual, individualToMutate),
            "Individual was never mutated."
        )
        
        #ensure mutation not outside of bounds
        self.assertTrue(
            all(
                trait <= upper_bound and trait >= lower_bound
                for trait in individualToMutate
            ),
            "A mutation made a trait go out of bounds."
        )
        
#Only things left to test are fitness graphs and data sorting
            
if __name__ == '__main__':
    unittest.main()