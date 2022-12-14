import copy
from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as ss
import anytree
from functools import reduce
import operator as OPER
import sklearn.model_selection as sms

def IF(ops):
    conditional, trueRslt, falseRslt = ops[0], ops[1], ops[2]
    
    #print(f"if({conditional}) then {trueRslt} else {falseRslt}")
    
    if conditional:
        return trueRslt
    else:
        return falseRslt
 
def SUBTRACT(ops):
    #print(f"{ops[0]} - {ops[1]}")
    
    return reduce(OPER.sub, ops)

def MULTIPLY(ops):
    #print(f"{ops[0]} * {ops[1]}")
    
    return reduce(OPER.mul, ops)
    
def DIVIDE(ops):
    """Protected divison

    Args:
        ops (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    #print(f"{ops[0]} / {ops[1]}")
    
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
    def __init__(self, initDepth: int, initType: InitType, NT: set, T: set, x, y, softCapNodeMax) -> None:
        #var init
        self.initType = initType
        self.initDepth = initDepth
        self.T = T
        self.NT = NT
        self.softCapNodeMax = softCapNodeMax
        
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
       
    def EvaluateFitness(self, x, y, applyParsimonyPressure = True):
        """
        Fitness evaluated through using RMSE (Root Mean Sqrd Error).
        Applies parsimony pressure by default.
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
        
        fitness = np.sqrt(sqrdSum/inputCount)
            
        nodeCount = self.GetNodeCount()
            
        #if applying pressure and enough nodes to apply fitness mod
        if applyParsimonyPressure and nodeCount > self.softCapNodeMax:
            #incr fitness by every additional node over the max
            fitness = fitness * (nodeCount / self.softCapNodeMax)
            
        self.fitness = fitness
       
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
        parent.value = float( parent.operator.funct(ops) )
       
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
    """
    Genetic Program with individuals as trees.
    """
    def __init__(self, populationSize: int, initDepth: int, NT: set, T: set, x_train, y_train, pairs_of_parents_elitism_saves, softCapNodeMax: int = 10, xrate: float = 1):
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
        self.population = [Individual(initDepth, InitType.FULL, NT, T, x_train, y_train, softCapNodeMax) for _ in range(int(self.populationSize/2))] + [Individual(initDepth, InitType.GROWTH, NT, T, x_train, y_train, softCapNodeMax) for _ in range(int(self.populationSize/2))] 
        
        #init fitness lists w/ starting pop's fitness vals
        self.avgFitness = [self.GetAvgFitness()]
        self.bestFitness = [self.GetBestFitness()]
        self.worstFitness = [self.GetWorstFitness()] 
        self.bestFitnessNodeCount = [self.GetBestFitnessNodeCount()]
        self.worstFitnessNodeCount = [self.GetWorstFitnessNodeCount()]
           
    def RunGen(self) -> None:
        #create new pop
        self.CreateNextGeneration()
        
        #store newly created pops fitness fields
        self.avgFitness.append( self.GetAvgFitness() ) 
        self.worstFitness.append( self.GetWorstFitness() ) 
        self.bestFitness.append( self.GetBestFitness() )
        self.bestFitnessNodeCount.append(self.GetBestFitnessNodeCount())
        self.worstFitnessNodeCount.append(self.GetWorstFitnessNodeCount())
        
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
   
    def GetBestFitIndividual(self) -> Individual:
        return min( self.population, key=getFitness )
    
    def GetWorstFitIndividual(self) -> Individual:
        return max( self.population, key=getFitness )
   
    def GetBestFitnessNodeCount(self) -> int:
        return self.GetBestFitIndividual().GetNodeCount()
    
    def GetWorstFitnessNodeCount(self) -> int:
        return self.GetWorstFitIndividual().GetNodeCount()
   
    def GetBestFitness(self) -> float:
        return self.GetBestFitIndividual().fitness
    
    def GetWorstFitness(self) -> float:
        return self.GetWorstFitIndividual().fitness
    
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
        plt.title('Generational Fitness Data')
        plt.plot(t, self.worstFitness, label='Worst Fitness') 
        plt.plot(t, self.avgFitness, label='Average Fitness') 
        plt.plot(t, self.bestFitness, label='Best Fitness') 
        plt.grid() 
        plt.legend()
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        
        #init worst worst fit
        worstWorstFitness = max(self.worstFitness)
        
        fitnessIndex = 0
        
        for fitnessData in (self.worstFitness, self.avgFitness, self.bestFitness):
            yAnnotatePosition = worstWorstFitness - worstWorstFitness * fitnessIndex / 12
            
            fitnessIndex += 1
            
            plt.annotate('%0.7f' % min(fitnessData), xy=(1, yAnnotatePosition), xytext=(8, 0), 
                        xycoords=('axes fraction', 'data'), textcoords='offset points')
        
        plt.show()
    
    
    def PlotGenerationalNodeCount(self):
        t = np.arange(0, self.currentGeneration+1)
            
        plt.rcParams.update({'font.size': 22})
        plt.plot(t, self.bestFitnessNodeCount, label='Best Fitness') 
        plt.plot(t, self.worstFitnessNodeCount, label='Worst Fitness')
        plt.grid() 
        plt.legend()
        plt.title('Node Count per Generation')
        plt.ylabel('Node Count')
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
        
        #sel best individual
        bestIndividual = self.GetBestFitIndividual()
        
        #print("Best Individual:")
        #print(anytree.RenderTree(bestIndividual.root))
        
        for i in range(inputCount):
            bestIndividual.EvaluateFitnessRecursively(bestIndividual.root, x[i])
            
            y_pred[i] = bestIndividual.root.value
            
        return y_pred
        
    def Test(self, x, y, training_bounds: tuple):
        
        inputCount = len(x)
        
        assert inputCount == len(y)
        
        #for i in range(self.populationSize): #why re-eval fitness for new data??
        #    self.population[i].EvaluateFitness(x, y)
            
        bestFitIndividual = self.GetBestFitIndividual()
        
        print("Best fit individual:")
        print(anytree.RenderTree(bestFitIndividual.root))
        
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
        plt.ylim(-20, 20)
        # draw a vertical line at the optimal input
        plt.axvline(x=training_bounds[0], ls='--', color='red')
        plt.axvline(x=training_bounds[1], ls='--', color='red')
        plt.text(training_bounds[0] + 0.1, 5,'training range') #(training_bounds[1] - training_bounds[0])/2
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
    }

# define optimal input value
#x_optima = 0.96609
#construct terminal set
T = {0, 1, 1.4, 3, 18, np.pi, 'x'}

if __name__ == '__main__': 
    POPULATION_SIZE = 100
    INDIVIDUALS_NUMBER_OF_TRAITS = 10
    GENERATIONS_PER_RUN = 300  
    PAIRS_OF_PARENTS_SAVED_FOR_ELITISM = 1
    XRATE = 0.95
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
    strt, stp = 0.0, 1.2
    # define optimal input value
    x_optima = 0.96609
    # sample input range uniformly at 0.01 increments
    inputs = np.arange(strt, stp, 0.01)
    # compute targets
    results = objective(inputs)
    
    #test individual class
    #individual1 = Individual(INIT_DEPTH, InitType.FULL, NT, T, inputs, results)
    #individual2 = Individual(INIT_DEPTH, InitType.GROWTH, NT, T, inputs, results)
    
    #X_train, X_test, y_train, y_test = sms.train_test_split(inputs, results, test_size=0.33)
    #X_train, X_test, y_train, y_test = inputs, inputs, results, results
    
    X_train, y_train = inputs, results
    
    start_time = time.time()
    
    for i in range(20):
    
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
        #x_validation = np.arange(0.9, r_max, 0.001)
        #y_validation = objective(x_validation)
        
        for _ in range(GENERATIONS_PER_RUN):
        
            gp.RunGen()

        #if first GP
        if( i == 0 ):
            #make it the best
            bestFitnessGP = copy.copy( gp.bestFitness[-1] )
            bestGP = gp
        #if not first GP
        else:
            #if curr gp performs better than prev best
            if(gp.bestFitness[-1] < bestFitnessGP):
                #establish new best fitness GP
                bestFitnessGP = copy.copy( gp.bestFitness[-1] )
                bestGP = gp
      
    print(f"Runtime took {time.time() - start_time} seconds.")
        
    bestGP.PlotGenerationalFitness()
    bestGP.PlotGenerationalNodeCount()
    
    #test using data outside of input range
    # define range for input
    r_min, r_max = -1, 5
    # sample input range uniformly at increments
    x_test = np.arange(r_min, r_max, 0.001)
    # compute targets
    y_test = objective(x_test)
    #compare to test set outside of bounds
    bestGP.Test(x_test, y_test, training_bounds = (strt, stp))
    
    """
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
    """