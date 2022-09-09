import random

class gene:
    def __inti__(self):
        self.row = random.randint(1,8)
        self.col = random.randint(1,8)

    def mutate(self):
        if random.random() < 0.5:
            self.row += random.randrange(0,3) - 1
        else:
            self.col += random.randrange(0,3) - 1
        if self.row < 1:
            self.row = 1
        if self.row > 8:
            self.row = 8
        if self.col < 1:
            self.col = 1
        if self.col > 8:
            self.col = 8
    #def __
    

#test rep and mutation:
g = gene()
print(g)
g.mutate()
print(g)

def crossover(parent1, parent2):
    """Single point crossover.

    Args:
        parent1 (_type_): _description_
        parent2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    xpoint = random.randrange(1,7) #now using as index (don't want at very beginning or end bc don't wanna copy parents)
    c1 = parent1[:xpoint]+parent2[xpoint:] #concat of lists (0 to xpoint, xpoint to 0)
    c2 = parent1[:xpoint]+parent2[xpoint:]
    return c1, c2

p1 = [gene for i in range(8)]
p2 = [gene for i in range(8)]

print(p1)
print(p2)
c1,c2 = crossover(p1,p2)
print(c1)
print(c2)

def calcFitness(ind):
    score = 0
    for i,g in enumerate(ind): #index and gene in each individual (enumerate to sequence an object)
        for i2, g2 in enumerate(ind):
            if i != i2:
                if g.row == g2.row or g.col == g2.col:
                    score += 1
                #if abs val of slope = 1

#should move queen to pos it wouldn't be conflicting with any other queen + eval fitness
# could plug in actual solution
            