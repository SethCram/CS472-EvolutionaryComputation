import copy
from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy
import multiprocessing
import time
import scipy.stats as ss
import anytree

class InitType(Enum):
    GROWTH = 0
    FULL = 1

class Operator():
    def __init__(self, funct, arity) -> None:
        self.funct = funct
        self.arity = arity
        
class Individual():
    def __init__(self, initDepth: int, initType: InitType, NT: set, T: set) -> None:
        #var init
        self.initType = initType
        self.initDepth = initDepth
        self.T = T
        self.NT = NT
        self.fitness = 0
        #funct init
        self.tree = self.CreateTree()
       
    def CreateTree(self) -> list:
        """Adds a root non-terminal by default.

        Args:
            initType (InitType): _description_

        Returns:
            list: Tree nodes
        """
        nodes = []
        childrenNodes = []
        layerNodes = []
        
        layerNodes.append( anytree.Node( "0", operator=random.choice(tuple(NT)) ) )
        
        #walk thru each horizontal layer of tree, starting at depth 1 of root
        for currDepth in range(1, self.initDepth):
            
            #walk thru this layer's nodes
            for i in range(len(layerNodes)):
                #if layer node is a NT bc it has an operator field
                if hasattr(layerNodes[i], "operator"):
                    #for every member of its arity
                    for j in range(layerNodes[i].operator.arity):
                        #roll for the chance to create a NT or T node?
                        
                        #uniquely name node
                        #nodeName = f"depth: {currDepth+1}, parent: {i}, child: {j}"
                        nodeName = currDepth + i + j
                        nodeName = str(nodeName)
                        
                        #specify parent as curr layer node
                        #parentNode = layerNodes[i]
                        
                        #if layer right before last layer, so creating last layer
                        if currDepth == self.initDepth - 1:
                            #create a T child node
                            childrenNodes.append( 
                                anytree.Node(nodeName, 
                                operand=random.choice(tuple(self.T)), 
                                parent=layerNodes[i]) #specify parent as curr layer node
                            )
                        #if not creating last layer
                        else:
                            if self.initType == InitType.FULL:
                                #create a NT child node
                                childrenNodes.append( 
                                    anytree.Node(nodeName, 
                                    operator=random.choice(tuple(self.NT)), 
                                    parent=layerNodes[i]) #specify parent as curr layer node
                                )
                            elif self.initType == InitType.GROWTH:
                                #roll a 50/50 on whether child is T or NT
                                if numpy.random.randint(0,2) == 0:
                                     #create a NT child node
                                    childrenNodes.append( 
                                        anytree.Node(nodeName, 
                                        operator=random.choice(tuple(self.NT)), 
                                        parent=layerNodes[i]) #specify parent as curr layer node
                                    )
                                else:
                                    #create a T child node
                                    childrenNodes.append( 
                                        anytree.Node(nodeName, 
                                        operand=random.choice(tuple(self.T)), 
                                        parent=layerNodes[i]) #specify parent as curr layer node
                                    )
            #add layer nodes to overall nodes before overwrite
            nodes = copy.deepcopy(nodes) + copy.deepcopy(layerNodes) #+ copy.deepcopy(childrenNodes)
            #update layer nodes to newly created children nodes
            layerNodes = []
            layerNodes = copy.deepcopy(childrenNodes)
            #reset children nodes
            childrenNodes = []
        
        #copy over the last layer's nodes too
        nodes = nodes + copy.deepcopy(layerNodes)
        
        for pre, fill, node in anytree.RenderTree(nodes[0]):
            if hasattr(node, "operator"):
                print("%s%s %s %s" % (pre, node.name, fill, node.operator.funct))
            if hasattr(node, "operand"):
                print("%s%s %s %s" % (pre, node.name, fill, node.operand))
        
        print(f"Node count: {len(nodes)}")
        
        return nodes
       
    def RandomSelect(selSet: set):
        return selSet[numpy.random.randint(0, len(selSet))]
    def Mutate(self):
        pass
    
    def __str__(self):
        return f"{self.name}({self.age})"  

class Population():
    def __init__(self, populationSize: int, initDepth: int, mutationStdDev: float, NT: set, T: set) -> None:
        self.populationSize = populationSize
        self.avgFitness = 0
        self.bestFitness = 0
        self.worstFitness = 0
        
        #create individuals of half FULL and half GROWTH
        self.individuals = [Individual(initDepth, InitType.FULL, NT, T) for _ in range(int(self.populationSize/2))] + [Individual(initDepth, InitType.GROWTH, NT, T) for _ in range(int(self.populationSize/2))]
        
        for i in range(len(self.individuals)):
            self.individuals[i].fitness = self.EvaluateFitness(self.individuals[i]) 
        
    def EvaluateFitness(self, individual: Individual):
        return 0 #(x**2 + y - 11)**2 + (x + y**2 -7)**2
        
    def Crossover(self, parent1: Individual, parent2: Individual) -> tuple:
        pass
    
    def ParentSelection(self) -> tuple:
        xIndexRange, prob = self.SetupHalfNormIntDistr(self.populationSize, stdDev=30)
    
        #if overloaded to display distr graph
        if(False):
            #take randos using the calc'd prob and index range
            nums = numpy.random.choice(xIndexRange, size = 1000000, p = prob)
            #display distr histogram
            plt.rcParams.update({'font.size': 22})
            plt.hist(nums, bins = pop_size)
            plt.title("likelihood of each parent index being chosen")
            plt.ylabel("likelihood of being chosen")
            plt.xlabel("parent index")
            plt.show()
    
        parent1Index, parent2Index = int( numpy.random.choice(xIndexRange, size = 2, p = prob) )
        
        #make sure indices within array range
        assert parent1Index < self.populationSize and parent2Index < self.populationSize and type(parent1Index) == int and type(parent2Index) == int
    
        return self.population[parent1Index], self.population[parent2Index]
    
    def SetupHalfNormIntDistr(pop_size: int, stdDev: int) -> tuple:
        """
        The half normal integer distribution parent indices are drawn from.

        Returns:
            tuple: index range and probability funct
        """
        #take interval 1-100
        x = numpy.arange(1, pop_size+1) #bc upper bound is exclusive
        #store every number's +/-0.5
        xU, xL = x + 0.5, x - 0.5 
        #determine probability
        prob = ss.halfnorm.cdf(xU, scale = stdDev) - ss.halfnorm.cdf(xL, scale = stdDev) #scale represents inner quartiles
        prob = prob / prob.sum() # normalize the probabilities so their sum is 1
        #decr by 1 to find the index 0-99
        xIndexRange = x - 1
    
        return xIndexRange, prob
    
    def __str__(self):
        return f"{self.name}({self.age})"

class GP():
    def __init__(self, populationSize: int, initDepth: int, mutationStdDev: float, NT: set, T: set):
        self.populationSize = populationSize
        self.mutationStdDev = mutationStdDev
        #self.crossoverType
        #self.selectionType
        self.CurrentGeneration = 0
        
        self.avgFitness = []
        self.bestFitness = []
        self.worstFitness = []
        
        self.population = Population(populationSize, initDepth, 0.2, NT, T)
            
    def runGen(self) -> Population:
        pass
    
    def __str__(self):
        return f"{self.benchmark_funct} with pop size ({self.popSize})"

if __name__ == '__main__': 
    NT = {
        Operator(funct=numpy.add, arity=2), 
        Operator(funct=numpy.subtract, arity=2), 
        Operator(funct=numpy.multiply, arity=2), 
        Operator(funct=numpy.divide, arity=2), #division by zero always yields zero in integer arithmetic.
        Operator(funct=numpy.abs, arity=1),
    }
    
    T = {1,2,3}
    POPULATION_SIZE = 100
    INDIVIDUALS_NUMBER_OF_TRAITS = 10
    GENERATIONS_PER_RUN = 300  
    TRAIT_CHANGE_PERCENTAGE = 3
    PAIRS_OF_PARENTS_SAVED_FOR_ELITISM = 1
    NUMBER_OF_ISLANDS = 5
    MIGRATION_INTERVAL = 5
    PAIRS_OF_IMMIGRANTS = 3
    MAX_CHILDREN_KILLED = 10
    CROWDING_DIST_THRESH = 1
    CROWDING_NICHES = 30
    FITNESS_SHARING_RANGE = 1
    
    INIT_DEPTH = 3
    
    GP(
        populationSize=POPULATION_SIZE,
        initDepth=INIT_DEPTH,
        mutationStdDev=0.2,
        NT=NT,
        T=T
    )
    
    """
    nodes = []
    childrenNodes = []
    layerNodes = []
    
    layerNodes.append( anytree.Node( "0", operator=random.choice(tuple(NT)) ) )
    
    #walk thru each horizontal layer of tree, starting at depth 1 of root
    for currDepth in range(1, initDepth):
        
        #walk thru this layer's nodes
        for i in range(len(layerNodes)):
            #if layer node is a NT bc it has an operator field
            if hasattr(layerNodes[i], "operator"):
                #for every member of its arity
                for j in range(layerNodes[i].operator.arity):
                    #roll for the chance to create a NT or T node?
                    
                    #uniquely name node
                    #nodeName = f"depth: {currDepth+1}, parent: {i}, child: {j}"
                    nodeName = currDepth + i + j
                    nodeName = str(nodeName)
                    
                    #specify parent as curr layer node
                    #parentNode = layerNodes[i]
                    
                    #if layer right before last layer, so creating last layer
                    if currDepth == initDepth - 1:
                        #create a T child node
                        childrenNodes.append( 
                            anytree.Node(nodeName, 
                            operand=random.choice(tuple(T)), 
                            parent=layerNodes[i]) #specify parent as curr layer node
                        )
                    else:
                        #create a NT child node
                        childrenNodes.append( 
                            anytree.Node(nodeName, 
                            operator=random.choice(tuple(NT)), 
                            parent=layerNodes[i]) #specify parent as curr layer node
                        )
        #add layer nodes to overall nodes before overwrite
        nodes = copy.deepcopy(nodes) + copy.deepcopy(layerNodes) #+ copy.deepcopy(childrenNodes)
        #update layer nodes to newly created children nodes
        layerNodes = []
        layerNodes = copy.deepcopy(childrenNodes)
        #reset children nodes
        childrenNodes = []
    
    #copy over the last layer's nodes too
    nodes = nodes + copy.deepcopy(layerNodes)
    
    for pre, fill, node in anytree.RenderTree(nodes[0]):
        if hasattr(node, "operator"):
            print("%s%s %s" % (pre, node.name, node.operator.funct))
        if hasattr(node, "operand"):
            print("%s%s %s" % (pre, node.name, node.operand))
    
    print(len(nodes))
    print(nodes)
    """