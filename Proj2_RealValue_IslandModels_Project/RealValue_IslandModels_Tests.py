from functools import update_wrapper
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

    def test_EvalFitness(self):
        """
        Test to make sure an individual with the target input as traits 
        results in a fitness of zero for every function.
        """
        
        #walk thru funct bounds dict
        for functionEnum, functionTargetInput in functionInputTargetDict.items():
            
            testIndividual = numpy.full(INDIVIDUALS_NUMBER_OF_TRAITS, functionTargetInput, dtype=float)
            
            testIndividualFitness = EvalFitness(functionEnum, testIndividual)
            
            errorMsg = "When evaluated with an individual of all {}, {} resulted in a fitness of {}, not zero.".format(functionTargetInput, functionEnum, testIndividualFitness)
            
            self.assertEqual( 
                testIndividualFitness,
                0,
                errorMsg
            )

if __name__ == '__main__':
    unittest.main()