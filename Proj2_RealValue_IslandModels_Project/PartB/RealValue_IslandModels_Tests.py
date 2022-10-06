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
                num_of_traits=Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, 
                lower_bound_inclusive=lower_bound,
                upper_bound_inclusive=upper_bound
            )
            
            #trait number check
            self.assertTrue(
                len(testIndividual) == Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                "A test trait for {} doesn't have {} traits.".format(functionEnum, Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS) 
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
                population_size=Implementation_Consts.POPULATION_SIZE, 
                individuals_num_of_traits=Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS
            )
            
            newPopSize = len(population)
            
            #pop size check
            self.assertTrue(
                newPopSize == Implementation_Consts.POPULATION_SIZE,
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
            
            #testIndividual = numpy.full(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, functionTargetInput, dtype=float)
            testIndividual = numpy.full(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, functionTargetInput, dtype=numpy.float64)
            
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
        
        xIndexRange, prob = SetupHalfNormIntDistr(Implementation_Consts.POPULATION_SIZE, stdDev=30)
        
        #take rando nums using the calc'd prob and index range
        randoNums = numpy.random.choice(xIndexRange, size = testDataCount, p = prob)

        #get count of each unique num gen'd
        unique, counts = numpy.unique(randoNums, return_counts=True)
        
        #pair count of each index in a dict
        indexCountDict = dict(zip(unique, counts))
        
        compIndex1 = 0
        compIndex2 = int(Implementation_Consts.POPULATION_SIZE/3)
        
        #ensure index 0 occured more often than the comp2 index
        self.assertGreater(
            indexCountDict[compIndex1],
            indexCountDict[compIndex2],
            "A lower index in the distribution should usually occure more often."
        )

    def test_ParentSelection(self):
        """
        Tests that all parents generated by the selection function 
        come from the population. 
        #If told to display the distribution graph the selection function displays a plot.
        """
        
        for functionEnum, functionBounds in functionBoundsDict.items():
        
            #create new population
            population = CreatePopulation(
                functionBounds=functionBounds, 
                population_size=Implementation_Consts.POPULATION_SIZE, 
                individuals_num_of_traits=Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS
            )

            sols = set()

            #walk thru each individual in pop
            populationFitness = CreateLocalPopulationFitness(functionEnum, population, sols)
            
            #get parents from pop-fitness obj    
            parents = ParentSelection(populationFitness)
                        
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
            #ParentSelection(populationFitness, displayDistributionGraph=True)

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
          
    def test_ImmigrantSelection(self):
        """
        Tests that all immigrants generated by the selection function 
        come from the population.
        Also verifies desired number of immigrants were created.
        """
        for functionEnum, functionBounds in functionBoundsDict.items():
        
            #create new population
            population = CreatePopulation(
                functionBounds=functionBounds, 
                population_size=Implementation_Consts.POPULATION_SIZE, 
                individuals_num_of_traits=Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS
            )

            sols = set()

            #walk thru each individual in pop
            populationFitness = CreateLocalPopulationFitness(functionEnum, population, sols)
            
            #get immigrants from pop-fitness obj    
            immigrants = ImmigrantSelection(populationFitness, desiredImmigrants=Implementation_Consts.PAIRS_OF_IMMIGRANTS * 2)
            
            #verify number of immigrants desired was created
            self.assertEqual( len(immigrants), Implementation_Consts.PAIRS_OF_IMMIGRANTS * 2 )
                        
            #walk thru immigrants
            for migrant in immigrants:
                #make sure each migrant is actually in the pop
                self.assertTrue(
                    any((numpy.array_equal(migrant.individual, individual)) for individual in population),
                    "The selected immigrant {} isn't in the population.".format(migrant.individual)
                )
            
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
        parent1 = numpy.full(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, parent1Val)
        
        #init parent array w/ 1's
        parent2 = numpy.full(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, parent2Val)
        
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
                parent1Vals < Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS
                and parent2Vals < Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
                "This child is a clone of one of its parents."
            )
            
            #ensure all child traits from one of its parents
            self.assertTrue(
                parent1Vals + parent2Vals == Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS,
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
        
        individualToMutate = numpy.ones(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS)
        
        unMutatedIndividual = numpy.ones(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS)
        
        lower_bound = -1
        upper_bound = 1
        
        #loop till successful mutation
        while(
            Mutate((
                lower_bound, upper_bound), individualToMutate, 
                Implementation_Consts.TRAIT_CHANGE_PERCENTAGE
            ) == False):
            pass
    
        #ensure individual mutated
        self.assertFalse( 
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
        
    def test_SetupHalfNormIntDistr(self):
        """
        Tests the half normal integer distribution #parent indices are drawn from.
        #Verifies that a lower index number occures more often than a higher index number.
        """
        
        testDataCount = 1000000
        
        xIndexRange, prob = SetupHalfNormIntDistr(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, stdDev=1)
        
        #take rando nums using the calc'd prob and index range
        randoNums = numpy.random.choice(xIndexRange, size = testDataCount, p = prob)

        #get count of each unique num gen'd
        unique, counts = numpy.unique(randoNums, return_counts=True)
        
        #pair count of each index in a dict
        indexCountDict = dict(zip(unique, counts))
        
        compIndex1 = 0
        compIndex2 = int(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS/3)
        
        #ensure index 0 occured more often than the comp2 index
        self.assertGreater(
            indexCountDict[compIndex1],
            indexCountDict[compIndex2],
            "A lower index in the distribution should usually occure more often."
        )
        """
        #display distr histogram
        plt.rcParams.update({'font.size': 22})
        plt.hist(randoNums, bins = Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS)
        plt.title("Likelihood of each trait index being chosen")
        plt.ylabel("Occurences")
        plt.xlabel("Trait index")
        plt.show()
        """
        
#Only things left to test are fitness graphs and data sorting
         
class TestNichingRelated(unittest.TestCase):
    def test_Crowding(self):
        pass

    def test_DistanceTweenIndividuals(self):
        """
        Ensures a negative distance isn't generated 
        and that two of the same individuals are 0 distance apart.
        """
        
        individual1 = CreateRandomIndividual(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, 0, 1)
        individual2 = CreateRandomIndividual(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS, -1, 0)
        
        dist = GetDistTweenIndividuals(individual1, individual2)
        
        #make sure dist not negative
        self.assertLessEqual( 0, dist, "Distance can't be negative.")
        
        
        individual1 = numpy.zeros(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS)
        individual2 = numpy.zeros(Implementation_Consts.INDIVIDUALS_NUMBER_OF_TRAITS)
        
        dist = GetDistTweenIndividuals(individual1, individual2)
        
        #make sure dist is 0
        self.assertEqual( 0, dist, "Distance should be 0.")

    def test_FitnessSharing(self):
        pass
            
if __name__ == '__main__':
    unittest.main()