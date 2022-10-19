import copy
from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy
import multiprocessing
import time
import scipy.stats as ss
import anytree
from functools import reduce
import operator as OPER

def IF(ops):
    conditional, trueRslt, falseRslt = ops[0], ops[1], ops[2]
    if conditional:
        return trueRslt
    else:
        return falseRslt
 
def SUBTRACT(ops):
    return reduce(OPER.sub, ops)

def MULTIPLY(ops):
    return reduce(OPER.mul, ops)
    
def DIVIDE(ops):
    """Protected divison

    Args:
        ops (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    if( ops[1] == 0):
        return 0
    else:
        return ops[0] / ops[1]

class InitType(Enum):
    GROWTH = 0
    FULL = 1
    
class NodeType(Enum):
    TERMINAL = 0
    NONTERMINAL = 1
    
class Operator():
    def __init__(self, funct, arity) -> None:
        self.funct = funct
        self.arity = arity
        
    def __str__(self) -> str:
        return f"{self.funct}, arity {self.arity}"
    
    def __repr__(self) -> str:
        return f"{self.funct}, arity {self.arity}"
    
class Individual():
    def __init__(self, initDepth: int, initType: InitType, NT: set, T: set) -> None:
        #var init
        self.initType = initType
        self.initDepth = initDepth
        self.T = T
        self.NT = NT
        self.fitness = 0
        
        #funct init
        
        #initialize individual as tree
        self.root = self.CreateNodeNT(0, parent=None)
        self.nodeCount = 1
        self.CreateTreeRecursively(self.root)
        print(anytree.RenderTree(self.root))
        
        #fitness eval of tree
        self.EvaluateFitnessRecursively(self.root)
        self.fitness = self.root.value
       
    def EvaluateFitnessRecursively(self, parent: anytree.node):
        ops = []
        #walk thru children
        for child in parent.children:
            #if child is NT and not evaluated
            if (child.type == NodeType.NONTERMINAL and
                child.evaluated == False):
                #evaluate child
                self.EvaluateFitnessRecursively(child)
            
            ops.append(child.value)
        #evaluate parent using children values
        parent.value = parent.operator.funct(ops)
        parent.evaluated = True
       
    def EvaluateFitnessIteratively(self):
        """
        Evaluates the individual's tree.
        """
        
        tree_size = len(self.tree)
        
        #walk thru every node in tree
        for i in range(1, tree_size+1):
            #start walking up tree from bot
            currNode = self.tree[-i]
            
            operands = []
            
            #if T or NT that's already been eval'd
            if (currNode.type == NodeType.TERMINAL or 
                (currNode.type == NodeType.NONTERMINAL and currNode.evaluated == True)):
                #walk thru children parent of curr node
                for parentsChild in currNode.parent.children:
                    #if parentsChild = T
                    if( parentsChild.type == NodeType.TERMINAL ):
                        operands.append(parentsChild.operand)
                    #if parentsChild = NT and eval'd
                    elif( parentsChild.type == NodeType.NONTERMINAL and currNode.evaluated == True ):
                        operands.append(parentsChild.result)
                
                if currNode.parent.operator.arity == len(operands):
                    currNode.parent.result = currNode.parent.operator.funct(operands)
                    currNode.parent.evaluated = True
        
        #make sure tree been eval'd
        assert self.tree[0].evaluated == True    
        
        return self.tree[0].result
        
        #return (x**2 + y - 11)**2 + (x + y**2 -7)**2
       
    def CreateTreeRecursively(self, parent: anytree.Node) -> None:
            #every parent is a NT
            for _ in range(parent.operator.arity):
                self.nodeCount += 1
                nodeName = self.nodeCount
                
                #if creating laster layer of nodes
                if parent.depth == self.initDepth - 2: #depth starts at 0
                        #create T node
                        self.CreateNodeT(nodeName, parent) 
                #if not creating last layer
                else:
                    if self.initType == InitType.FULL:
                        #recursively create NT
                        self.CreateTreeRecursively( self.CreateNodeNT(nodeName, parent) )
                    elif self.initType == InitType.GROWTH:
                        #roll a 50/50 on whether child is T or NT
                        if numpy.random.randint(0,2) == 0:
                            #recursively create NT
                            self.CreateTreeRecursively( self.CreateNodeNT(nodeName, parent) )
                        else:
                            self.CreateNodeT(nodeName, parent)
       
    def CreateTreeIteratively(self) -> list:
        """Adds a root non-terminal by default.

        Args:
            initType (InitType): _description_

        Returns:
            list: Tree nodes
        """
        nodes = []
        childrenNodes = []
        layerNodes = []
        
        layerNodes.append( 
            self.CreateNodeNT("0", parent=None)
        )
        
        #walk thru each horizontal layer of tree, starting at depth 1 of root
        for currDepth in range(1, self.initDepth):
            
            #walk thru this layer's nodes
            for i in range(len(layerNodes)):
                #if layer node is a NT 
                if layerNodes[i].type == NodeType.NONTERMINAL: #hasattr(layerNodes[i], "operator"):
                    #for every member of its arity
                    for j in range(layerNodes[i].operator.arity):
                        #roll for the chance to create a NT or T node?
                        
                        #uniquely name node
                        #nodeName = f"depth: {currDepth+1}, parent: {i}, child: {j}"
                        nodeName = currDepth + i + j
                        nodeName = str(nodeName)
                        
                        #if layer right before last layer, so creating last layer
                        if currDepth == self.initDepth - 1:
                            #create a T child node
                            childrenNodes.append( 
                                self.CreateNodeT(nodeName, layerNodes[i]) #specify parent as curr layer node
                            )
                        #if not creating last layer
                        else:
                            if self.initType == InitType.FULL:
                                #create a NT child node
                                childrenNodes.append( 
                                    self.CreateNodeNT(nodeName, layerNodes[i])
                                )
                            elif self.initType == InitType.GROWTH:
                                #roll a 50/50 on whether child is T or NT
                                if numpy.random.randint(0,2) == 0:
                                    #create a NT child node
                                    childrenNodes.append( 
                                        self.CreateNodeNT(nodeName, layerNodes[i])
                                    )
                                else:
                                    #create a T child node
                                    childrenNodes.append( 
                                        self.CreateNodeT(nodeName, layerNodes[i])
                                    )
            #add layer nodes to overall nodes before overwrite
            nodes = copy.deepcopy(nodes) + copy.deepcopy(layerNodes) #+ copy.deepcopy(childrenNodes)
            #update layer nodes to newly created children nodes
            layerNodes = []
            layerNodes = copy.deepcopy(childrenNodes)
            #reset children nodes
            childrenNodes = []
        
        #copy over the last layer's nodes too
        #nodes = nodes + copy.deepcopy(layerNodes)
        
        #for pre, fill, node in anytree.RenderTree(nodes[0]):
        #    if hasattr(node, "operator"):
        #        print("%s%s %s %s" % (pre, node.name, fill, node.operator.funct))
        #    if hasattr(node, "operand"):
        #        print("%s%s %s %s" % (pre, node.name, fill, node.operand))
        #print(anytree.RenderTree(nodes[0], style=anytree.AsciiStyle(), maxlevel=4))
        #for node in nodes:
        #    print(anytree.RenderTree(node))
        #print(anytree.RenderTree(nodes[0]))
        #print(f"Node count: {len(nodes)}")
        
        #return nodes
        return nodes[0]
       
    def CreateNodeNT(self, nodeName, parent) -> anytree.Node:
        #create a NT child node
        return anytree.Node(nodeName, 
            operator=random.choice(tuple(self.NT)),
            type = NodeType.NONTERMINAL,  
            value = 0,
            evaluated = False,                                    
            parent = parent
        )
    
    def CreateNodeT(self, nodeName, parent) -> anytree.Node:
        #create a T child node
        return anytree.Node(nodeName, 
            value = random.choice(tuple(self.T)), 
            type = NodeType.TERMINAL, 
            parent = parent
        )
       
    #def Mutate(self):
    #    num_of_nodes = len(self.tree)
    
    def __str__(self):
        return f"{self.name}({self.age})"  

def getFitness( individual: Individual ) -> int:
    return individual.fitness

class GP():
    def __init__(self, populationSize: int, initDepth: int, NT: set, T: set):
        self.populationSize = populationSize
        self.initDepth = initDepth
        self.NT = NT
        self.T = T
        #self.crossoverType
        #self.selectionType
        self.currentGeneration = 0
        
        self.population = [Individual(initDepth, InitType.FULL, NT, T) for _ in range(int(self.populationSize/2))] + [Individual(initDepth, InitType.GROWTH, NT, T) for _ in range(int(self.populationSize/2))] #Population(self.populationSize, self.initDepth, self.NT, self.T)
        
        #init fitness lists w/ starting pop's fitness vals
        self.avgFitness = [self.GetAvgFitness()]
        self.bestFitness = [self.GetBestFitness()]
        self.worstFitness = [self.GetWorstFitness()] 
           
    def RunGen(self) -> None:
        #create new pop
        self.CreateNextGeneration()
        
        #store newly created pops fitness fields
        self.avgFitness.append( self.GetAvgFitness() ) 
        self.worstFitness.append( self.GetWorstFitness() ) 
        self.bestFitness.append( self.GetBestFitness() )
        
        #advance gen count
        self.currentGeneration += 1
   
    def CreateNextGeneration(self) ->None:
        #ensure individuals sorted in ascending order
        self.OrderPopulationByFitness()
        #make sure new pop empty
        newPopulation = []
        
        #walk thru half pop
        for _ in range(int(self.populationSize/2)):
            #select parents
            parent1, parent2 = self.SelectParents()
            #do crossover
            child1, child2 = self.Crossover(parent1, parent2)
            #add new children to next gen pop
            newPopulation.append(child1)
            newPopulation.append(child2)
            
        #don't needa deep copy bc newPopulation wiped out w/ leave funct
        self.population = newPopulation
   
    def GetBestFitness(self) -> float:
        return max( self.population, key=getFitness ).fitness
    
    def GetWorstFitness(self) -> float:
        return sum(self.population, key=getFitness ) / self.populationSize
    
    def GetAvgFitness(self) -> float:
        return min( self.population, key=getFitness ).fitness
    
    def Crossover(self, parent1: Individual, parent2: Individual) -> tuple:
        """80% of time NT crossover, 20% of time T crossover. 
        Never chooses the root node to do crossover with.

        Args:
            parent1 (Individual): _description_
            parent2 (Individual): _description_

        Returns:
            tuple: _description_
        """
        
        #pick crossover points
        p1_xpoint = numpy.random.randint(1, parent1.shape[0])
        p2_xpoint = numpy.random.randint(1, parent2.shape[0])
        
        #if rand is 80% of time
        if( numpy.random.randint(1,11) <= 8):
            
            #while they're both terminals
            while( 
                parent1.tree[p1_xpoint].type == NodeType.TERMINAL and 
                parent2.tree[p2_xpoint].type == NodeType.TERMINAL 
            ):
                #pick new crossover point for both
                p1_xpoint = numpy.random.randint(1, parent1.shape[0])
                p2_xpoint = numpy.random.randint(1, parent2.shape[0])
            
            #now atleast one is a NT
        
        #if rand is 20% of time
        else:
            #while they're both NTs
            while( 
                parent1.tree[p1_xpoint].type == NodeType.NONTERMINAL and 
                parent2.tree[p2_xpoint].type == NodeType.NONTERMINAL 
            ):
                #pick new crossover point for both
                p1_xpoint = numpy.random.randint(1, parent1.shape[0])
                p2_xpoint = numpy.random.randint(1, parent2.shape[0])
                
            #now atleast one is a T
        
        #if( parent1.tree[p1_xpoint].is_root == True ):
            
        #copy each child's parent
        child1 = copy.deepcopy(parent1) #.tree[0:p1_xpoint]
        child2 = copy.deepcopy(parent2) #.tree[0:p2_xpoint]
        
        #create a ph
        c1_parent_nodes = copy.deepcopy( child1.tree[p1_xpoint].parent )
        #assign each crossover point the over child's parent
        child1.tree[p1_xpoint].parent == copy.deepcopy( child2.tree[p2_xpoint].parent )
        child2.tree[p2_xpoint].parent == copy.deepcopy( c1_parent_nodes )
        
        #crossover at crosover point
        child1.tree = child1.tree[0:p1_xpoint-1] + child2.tree[p2_xpoint:]
        child2.tree = child2.tree[0:p2_xpoint-1] + child1.tree[p1_xpoint:]
        
        return child1, child2
    
    def SelectParents(self) -> tuple:
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
    
    def SetupHalfNormIntDistr(self, pop_size: int, stdDev: int) -> tuple:
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
    
    def OrderPopulationByFitness(self):
        #sort in descending order
        self.population.sort(key=getFitness)
    
    def PlotGenerationalFitness(self):
        t = numpy.arange(0, self.currentGeneration+1)
            
        plt.rcParams.update({'font.size': 22})
        plt.plot(t, self.bestFitness) 
        plt.grid() 
        plt.title('Best Fitness per Generation')
        plt.ylabel('Best Fitness')
        plt.xlabel('Generation')
        plt.show()

        #plt.subplot(3, 1, 2)
        plt.plot(t, self.avgFitness) 
        plt.grid() 
        plt.title('Average Fitness per Generation')
        plt.ylabel('Average Fitness')
        plt.xlabel('Generation')
        plt.show()

        #plt.subplot(3, 1, 3)
        plt.plot(t, self.worstFitness) 
        plt.grid() 
        plt.title('Worst Fitness per Generation')
        plt.ylabel('Worst Fitness')
        plt.xlabel('Generation')
        plt.show()
    
    def __str__(self):
        return f"{self.benchmark_funct} with pop size ({self.popSize})"

NT = {
        Operator(funct=sum, arity=2), 
        Operator(funct=SUBTRACT, arity=2), 
        Operator(funct=MULTIPLY, arity=2), 
        Operator(funct=DIVIDE, arity=2), #division by zero always yields zero in integer arithmetic.
        Operator(funct=numpy.abs, arity=1),
        Operator(funct=IF, arity=3),
    }
    
T = {1,2,3}

if __name__ == '__main__': 
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
    
    INIT_DEPTH = 4
    
    
    for i in range(INIT_DEPTH):
        if(i == 0):
            nodes = [anytree.Node(0)]
        else:
            nodes.append(anytree.Node(
                i, parent = nodes[i-1]
            ))
    
    print(anytree.RenderTree(nodes[0]))
    
    nodes[3].parent 
    
    print(anytree.RenderTree(nodes[0]))
    
    #test individual class
    individual1 = Individual(INIT_DEPTH, InitType.FULL, NT, T)
    individual2 = Individual(INIT_DEPTH, InitType.GROWTH, NT, T)
    
    #test GP
    treeGP = GP(
        populationSize=POPULATION_SIZE,
        initDepth=INIT_DEPTH,
        NT=NT,
        T=T
    )
    
    for _ in range(GENERATIONS_PER_RUN):
    
        treeGP.RunGen()