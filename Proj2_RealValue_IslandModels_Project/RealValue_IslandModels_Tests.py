from functools import update_wrapper
import unittest

from RealValue_IslandModels_Lib import *

class TestStringMethods(unittest.TestCase):

    def test_IndividualCreation(self):
        """
        Tests individual creation function once for every function.
        Ensures each generated trait is within the function's domain.
        """
        
        #for function in GA_Functions:
        
        #walk thru funct bounds
        for functionEnum, functionBounds in functionBoundsDict:
            
            #store bounds
            lower_bound, upper_bound = functionBounds
            #create rando individual
            
            testIndividual = CreateRandomIndividual(
                num_of_traits=INDIVIDUALS_NUMBER_OF_TRAITS, 
                lower_bound_inclusive=lower_bound,
                upper_bound_inclusive=upper_bound
            )
            
            #walk thru individual's traits
            for testTrait in testIndividual:
                self.assertTrue( 
                    testTrait >= lower_bound and testTrait <= upper_bound, 
                    "Test trait for {} is out of bounds.".format(functionEnum) 
                )

    def test_PopulationCreation(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()