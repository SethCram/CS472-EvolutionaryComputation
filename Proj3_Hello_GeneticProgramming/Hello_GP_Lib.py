import copy
from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import time
import scipy.stats as ss
import anytree
from functools import reduce
import operator as OPER
import sklearn.model_selection as sms

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
    
# objective function
def objective(x):
	return -(1.4 - 3.0 * x) * np.sin(18.0 * x)
    
class Individual():
    def __init__(self, initDepth: int, initType: InitType, NT: set, T: set, x, y) -> None:
        #var init
        self.initType = initType
        self.initDepth = initDepth
        self.T = T
        self.NT = NT
        
        assert len(x) == len(y)
        
        #funct init
        
        #initialize individual as tree
        self.nodeIndex = 0
        self.root = self.CreateNodeNT(self.nodeIndex, parent=None)
        self.CreateTreeRecursively(self.root)
        #print(anytree.RenderTree(self.root))
        
        assert self.nodeIndex == self.GetNodeCount() - 1
        #print(f"self node count = {self.nodeCount}, get node count = {self.GetNodeCount()}")
        
        #fitness eval of tree
        self.EvaluateFitness(x, y)
       
    def EvaluateFitness(self, x, y):
        """
        Fitness evaluated through using RMSE (Root Mean Sqrd Error).
        Lower fitness is better.
        """
        
        sqrdSum = 0
        
        inputCount = len(x)
        
        assert inputCount == len(y)
        
        for i in range(inputCount):
            #NT's assigned values
            self.EvaluateFitnessRecursively(self.root, x[i])
            #accrue sqrd error
            sqrdSum += ( y[i] - float( self.root.value ) )**2
            
        self.fitness = np.sqrt(sqrdSum/inputCount)
       
    def EvaluateFitnessRecursively(self, parent: anytree.node, x: float):
        """NT nodes assigned values.

        Args:
            parent (anytree.node): _description_
        """
        ops = []
        #walk thru children
        for child in parent.children:
            #if child is NT and not evaluated
            if (child.type == NodeType.NONTERMINAL):
                #evaluate child (don't change input val)
                self.EvaluateFitnessRecursively(child, x)
            #if child val is a variable 
            if( child.value == 'x'):
                #substitute passed in var
                ops.append(x)
            #regular child 
            else:
                ops.append(child.value)
        
        #evaluate parent using children values
        parent.value = parent.operator.funct(ops)
       
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
                self.nodeIndex += 1
                nodeName = self.nodeIndex
                
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
                        if np.random.randint(0,2) == 0:
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
                                if np.random.randint(0,2) == 0:
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
            parent = parent
        )
    
    def CreateNodeT(self, nodeName, parent) -> anytree.Node:
        #create a T child node
        return anytree.Node(nodeName, 
            value = random.choice(tuple(self.T)), 
            type = NodeType.TERMINAL, 
            parent = parent
        )
       
    def GetNodeCount(self) -> int:
        """Calcs node count through counting the root's descendants.
        Needs to dynamically calculate node count bc during crossover, tree size changes.

        Returns:
            int: _description_
        """
        #return number of descendants and 1 to account for root
        return len(self.root.descendants) + 1
       
    #def Mutate(self):
    #    num_of_nodes = len(self.tree)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            # don't attempt to compare against unrelated types
            return NotImplemented
        
        return self.GetNodeCount() == other.GetNodeCount() #should iteratively compare each node
    
    def __str__(self):
        return f"{anytree.RenderTree(self.root)}, fitness of {self.fitness}"  

def getFitness( individual: Individual ) -> int:
    return individual.fitness

class GP():
    def __init__(self, populationSize: int, initDepth: int, NT: set, T: set, x_train, y_train, pairs_of_parents_elitism_saves, xrate: float = 1):
        self.populationSize = populationSize
        self.initDepth = initDepth
        self.NT = NT
        self.T = T
        self.xrate = xrate
        self.x_train = x_train
        self.y_train = y_train
        #self.selectionType
        self.currentGeneration = 0
        self.pairs_of_parents_elitism_saves = pairs_of_parents_elitism_saves
        
        #create pop of 50/50 growth/full individuals
        self.population = [Individual(initDepth, InitType.FULL, NT, T, x_train, y_train) for _ in range(int(self.populationSize/2))] + [Individual(initDepth, InitType.GROWTH, NT, T, x_train, y_train) for _ in range(int(self.populationSize/2))] 
        
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
        #new pop
        newPopulation = []
        
        #Save parents for elitism 
        for k in range(0, self.pairs_of_parents_elitism_saves):
            newPopulation.append(self.population[k])
            newPopulation.append(self.population[k+1])
        
        pairs_of_children = int(self.populationSize/2)
        
        #walk thru half pop
        for _ in range(self.pairs_of_parents_elitism_saves, pairs_of_children):
            #select parents
            parent1, parent2 = self.SelectParents()
            #do crossover
            child1, child2, xover = self.Crossover(parent1, parent2, self.xrate)
            #if crossover happened
            if(xover):
                #re'eval children fitness
                child1.EvaluateFitness(self.x_train, self.y_train)
                child2.EvaluateFitness(self.x_train, self.y_train)
            #add new children to next gen pop
            newPopulation.append(child1)
            newPopulation.append(child2)
            
        #don't needa deep copy bc newPopulation wiped out w/ leave funct
        self.population = newPopulation
   
    def GetBestFitness(self) -> float:
        return min( self.population, key=getFitness ).fitness
    
    def GetWorstFitness(self) -> float:
        return max( self.population, key=getFitness ).fitness
    
    def GetAvgFitness(self) -> float:
        fitnessSum = 0
        for i in range(0, self.populationSize):
            #take the fitness sum
            fitnessSum +=  self.population[i].fitness
        
        return fitnessSum / self.populationSize
    
    def Crossover(self, parent1: Individual, parent2: Individual, xrate: float = 1) -> tuple:
        """Swaps subtree parents at their xpoints. 
        Xpoints gauss centered around last leaf.
        Never chooses the root node to do crossover with.

        Args:
            parent1 (Individual): _description_
            parent2 (Individual): _description_

        Returns:
            tuple: child1, child2, whether xover happened
        """
        
        #clone children from parents
        child1 = copy.deepcopy(parent1) 
        child2 = copy.deepcopy(parent2)
        
        #roll on whether to do crossover
        randProb = np.random.random()
        xover = randProb <= xrate
        if( xover ):
        
            #pick crossover subtress
            parent1subtree, parent2subtree = self.GetCrossoverSubtrees(child1, child2)
        
            """
            #if rand is 80% of time
            if( np.random.randint(1,11) <= 8):
                
                #while they're both terminals
                while( 
                    parent1subtree.type  == NodeType.TERMINAL and 
                    parent2subtree.type  == NodeType.TERMINAL 
                ):
                    #pick new crossover point for both
                    parent1subtree, parent2subtree = self.GetCrossoverSubtrees(child1, child2)
                
                #now atleast one is a NT
            
            #if rand is 20% of time
            else:
                #while they're both NTs
                while( 
                    parent1subtree.type == NodeType.NONTERMINAL and 
                    parent2subtree.type == NodeType.NONTERMINAL 
                ):
                    #pick new crossover point for both
                    parent1subtree, parent2subtree = self.GetCrossoverSubtrees(child1, child2)
                    
                #now atleast one is a T
            """
            
            #swap subtree parents (don't copy)
            parent1subtree_parent_ph = parent1subtree.parent 
            #print(anytree.RenderTree(child1.root))
            #print(anytree.RenderTree(child2.root))
            parent1subtree.parent = parent2subtree.parent
            parent2subtree.parent = parent1subtree_parent_ph
            #print(anytree.RenderTree(child1.root))
            #print(anytree.RenderTree(child2.root))

        return child1, child2, xover
    
    def GetCrossoverSubtrees(self, parent1, parent2) -> tuple:
        """Swaps subtrees at last leaf gauss random indices.

        Args:
            parent1 (_type_): _description_
            parent2 (_type_): _description_

        Returns:
            tuple: child1, child2 still connected to parent1 and parent2 (not copies)
        """
        
        #use whatever parent has less nodes to choose xpoint
        #if( parent1.GetNodeCount() < parent2.GetNodeCount()):
        #    xpointUpperBound = parent1.GetNodeCount()
        #else:
        #    xpointUpperBound = parent2.GetNodeCount()
        #pick crossover points
        #p1_xpoint, p2_xpoint = np.random.randint(1, xpointUpperBound, size=2)
        #parent1subtree = anytree.find(parent1.root, filter_= lambda node: node.name == p1_xpoint) #could also index root descendants instead
        #parent2subtree = anytree.find(parent2.root, filter_= lambda node: node.name == p2_xpoint)
        
        #cache parent node counts
        p1Nodes = parent1.GetNodeCount()
        p2Nodes = parent2.GetNodeCount()
        #find descendant node count
        p1descendantNodes = p1Nodes - 1
        p2descendantNodes = p2Nodes - 1
        
        #gen half-normal range of ints centered at 0
        # std dev of 1/4 of descendant nodes count
        p1xIndexRange, p1prob = self.SetupHalfNormIntDistr(p1descendantNodes, stdDev=p1descendantNodes/4)
        p2xIndexRange, p2prob = self.SetupHalfNormIntDistr(p2descendantNodes, stdDev=p2descendantNodes/4)
        
        #p1_xpoint = int( np.random.randint(0, p1Nodes-1, size=1) )
        #p2_xpoint = int ( np.random.randint(0, p2Nodes-1, size=1) )
        #parent1subtree = parent1.root.descendants[p1_xpoint]
        #parent2subtree = parent2.root.descendants[p2_xpoint]
        
        #sel parent xpoints from 1 to descendant nodes count
        p1_xpoint = int( np.random.choice(p1xIndexRange+1, size = 1, p = p1prob) )
        p2_xpoint = int( np.random.choice(p2xIndexRange+1, size = 1, p = p2prob) )
        
        #apply xpoint, starting from the end
        # so norm distr centered around end of list (more terminals, smaller NTs)
        parent1subtree = parent1.root.descendants[-p1_xpoint]
        parent2subtree = parent2.root.descendants[-p2_xpoint]
        
        #debug: print(f"Crossover at {parent1subtree.name} and {parent2subtree.name}")
        
        assert parent1subtree != None, f"Couldn't find a node with xpoint {-p1_xpoint-1} in tree {anytree.RenderTree(parent1.root)}"
        assert parent2subtree != None, f"Couldn't find a node with xpoint {-p2_xpoint-1} in tree {anytree.RenderTree(parent2.root)}"
        
        return parent1subtree, parent2subtree
    
    def SelectParents(self) -> tuple:
        xIndexRange, prob = self.SetupHalfNormIntDistr(self.populationSize, stdDev=30)
    
        #if overloaded to display distr graph
        if(False):
            #take randos using the calc'd prob and index range
            nums = np.random.choice(xIndexRange, size = 1000000, p = prob)
            #display distr histogram
            plt.rcParams.update({'font.size': 22})
            plt.hist(nums, bins = pop_size)
            plt.title("likelihood of each parent index being chosen")
            plt.ylabel("likelihood of being chosen")
            plt.xlabel("parent index")
            plt.show()

        #get parent indices
        parent1Index, parent2Index = np.random.choice(xIndexRange, size = 2, p = prob)
        #parent1Index, parent2Index = parentIndices[0], parentIndices[1]
        
        #make sure indices within array range
        #assert parent1Index < self.populationSize and parent2Index < self.populationSize and type(parent1Index) == int and type(parent2Index) == int
    
        return self.population[int(parent1Index)], self.population[int(parent2Index)]
    
    def SetupHalfNormIntDistr(self, pop_size: int, stdDev: int) -> tuple:
        """
        The half normal integer distribution parent indices are drawn from.

        Returns:
            tuple: index range and probability funct
        """
        #take interval 1-100
        x = np.arange(1, pop_size+1) #bc upper bound is exclusive
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
        t = np.arange(0, self.currentGeneration+1)
            
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
    
    def Predict(self, x) -> list:
        """
        Predict using the best fit individual.
        """
        inputCount = len(x)
        
        y_pred = np.empty(inputCount)
        
        """
        y_pred = np.empty((self.populationSize, inputCount))
        
        for j in range(self.populationSize):
            for i in range(inputCount):
                self.population[j].EvaluateFitnessRecursively(self.root, x[i])
                
                y_pred[j][i] = self.population[j].root.value
        """
        #order pop by fitness 
        self.OrderPopulationByFitness()
        
        #sel best individual
        bestIndividual = self.population[0]
        
        for i in range(inputCount):
            bestIndividual.EvaluateFitnessRecursively(bestIndividual.root, x[i])
            
            y_pred[i] = bestIndividual.root.value
            
        return y_pred
        
    def Test(self, x, y):
        
        inputCount = len(x)
        
        assert inputCount == len(y)
        
        for i in range(self.populationSize):
            self.population[i].EvaluateFitness(x, y)
            
        bestFitness = self.GetBestFitness()
        
        print(f"Best fit individual = {bestFitness}")
        
        y_pred = self.Predict(x)
        
        assert len(y_pred) == len(y)
        
        #t = np.arange(0, inputCount)
            
        plt.rcParams.update({'font.size': 22})
        plt.plot(x, y_pred, label='Predictions') 
        plt.plot(x, y, label='Targets') 
        plt.legend()
        plt.grid() 
        plt.title('Predictions vs Targets')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()
    
    def __str__(self):
        return f"{self.population}, best fitness {self.bestFitness[-1]}"

NT = {
        Operator(funct=sum, arity=2), 
        Operator(funct=SUBTRACT, arity=2), 
        Operator(funct=MULTIPLY, arity=2), 
        Operator(funct=DIVIDE, arity=2), #division by zero always yields zero in integer arithmetic.
        Operator(funct=np.abs, arity=1),
        Operator(funct=IF, arity=3),
        Operator(funct=np.sin, arity=1),
        Operator(funct=np.cos, arity=1),
    }

# define optimal input value
#x_optima = 0.96609
#construct terminal set
T = {1.4, 3, 18, 'x'}

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
    XRATE = 0.8
    INIT_DEPTH = 4
    
    """
    for i in range(INIT_DEPTH):
        if(i == 0):
            nodes = [anytree.Node(f"t1_{i}")]
        else:
            nodes.append(anytree.Node(
                f"t1_{i}", parent = nodes[i-1]
            ))
            
    for i in range(INIT_DEPTH):
        if(i == 0):
            nodes2 = [anytree.Node(f"t2_{i}")]
        else:
            nodes2.append(anytree.Node(
                f"t2_{i}", parent = nodes2[i-1]
            ))
    
    
    print(anytree.RenderTree(nodes[0]))
    print(anytree.RenderTree(nodes2[0]))
    nodes2parent_ph = nodes[0].descendants[1].parent
    nodes[0].descendants[1].parent = nodes2[0].descendants[1].parent
    nodes2[0].descendants[1].parent = nodes2parent_ph
    #nodes[2].parent = nodes[1].parent
    print(anytree.RenderTree(nodes[0]))
    print(anytree.RenderTree(nodes2[0]))
    """
    
    # define range for input
    r_min, r_max = 0.0, 1.2
    # define optimal input value
    x_optima = 0.96609
    # sample input range uniformly at 0.01 increments
    inputs = np.arange(r_min, r_max, 0.01)
    # compute targets
    results = objective(inputs)
    
    #test individual class
    #individual1 = Individual(INIT_DEPTH, InitType.FULL, NT, T, inputs, results)
    #individual2 = Individual(INIT_DEPTH, InitType.GROWTH, NT, T, inputs, results)
    
    #X_train, X_test, y_train, y_test = sms.train_test_split(inputs, results, test_size=0.33)
    #X_train, X_test, y_train, y_test = inputs, inputs, results, results
    
    X_train, y_train = inputs, results
    
    #test GP
    gp = GP(
        populationSize=POPULATION_SIZE,
        initDepth=INIT_DEPTH,
        NT=NT,
        T=T,
        xrate=XRATE,
        x_train=X_train,
        y_train=y_train,
        pairs_of_parents_elitism_saves=PAIRS_OF_PARENTS_SAVED_FOR_ELITISM
    )
    
    #validation data # sample input range uniformly at 0.01 increments
    x_validation = np.arange(0.9, r_max, 0.001)
    y_validation = objective(x_validation)
    
    for _ in range(GENERATIONS_PER_RUN):
    
        gp.RunGen()
        
        #gp.PlotGenerationalFitness()
    
    #compare to validation set within bounds    
    gp.Test(x_validation, y_validation)
        
    gp.PlotGenerationalFitness()
    
    #test using data outside of input range
    # define range for input
    r_min, r_max = 1.3, 5
    # sample input range uniformly at 0.01 increments
    x_test_outside = np.arange(r_min, r_max, 0.01)
    # compute targets
    y_test_outside = objective(x_test_outside)
    #compare to test set outside of bounds
    gp.Test(x_test_outside, y_test_outside)
    
    
    # define range for input
    r_min, r_max = 0.0, 1.2
    # define optimal input value
    x_optima = 0.96609
    # sample input range uniformly at 0.01 increments
    inputs = np.arange(r_min, r_max, 0.01)
    # compute targets
    results = objective(inputs)
    # create a line plot of input vs result
    plt.plot(inputs, results)
    # draw a vertical line at the optimal input
    plt.axvline(x=x_optima, ls='--', color='red')
    
    # show the plot
    plt.show()