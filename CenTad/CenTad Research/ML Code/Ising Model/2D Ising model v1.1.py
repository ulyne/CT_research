import random

class IsingGrid:
    def __init__(self,dimension,N):
        self.dimension = dimension
        self.gridSize = N
        self.numSites = N**dimension
        self.MAX_E = dimension*N**dimension
        self.array = [0]*N**dimension
        self.fieldB = 0.0
        for i in range(0,N**dimension):
            self.array[i] = -1+2*int(random.random()*2)
    
    def getIndex(self,indexArray):
        index = 0;
        for i in range(0,self.dimension):
            index+=self.gridSize**(self.dimension - i - 1)*indexArray[self.dimension-i-1]
        return self.array[index]
    def flipIndex(self,indexArray):
        index = 0;
        for i in range(0,self.dimension):
            index+=self.gridSize**(self.dimension - i - 1)*indexArray[self.dimension-i-1]
        self.array[index]*=-1
    def neighborHalfEnergy(self,indexArray):
        sum=0.0
        val=self.getIndex(indexArray)
        dim = self.dimension
        N = self.gridSize
        for i in range(0,dim):
            ind = indexArray[0:i] + [ (indexArray[i] + 1) % N ] + indexArray[i+1:dim] #sum up each neigbor by adding one to each coordinate dimension
            sum+= -self.getIndex(ind)*val
        return sum
    def neighborEnergy(self,indexArray):
        sum=0.0
        val=self.getIndex(indexArray)
        dim = self.dimension
        N = self.gridSize
        for i in range(0,dim):
            ind1 = indexArray[0:i] + [ (indexArray[i] + 1) % N ] + indexArray[i+1:dim] #sum up each neigbor by adding one and subtracting one to each coordinate dimension
            ind2 = indexArray[0:i] + [ (indexArray[i] - 1) % N ] + indexArray[i+1:dim]
            sum+= -self.getIndex(ind1)*val
            sum+= -self.getIndex(ind2)*val
        return sum
    def calcEnergy(self,indexArrayPartial = []):
        if(len(indexArrayPartial)==self.dimension):
            return self.neighborHalfEnergy(indexArrayPartial) + self.getIndex(indexArrayPartial)*self.fieldB
        else:
            sum = 0.0
            for i in range(0,self.gridSize):
                sum += self.calcEnergy(indexArrayPartial+[i])
            return sum
    def deltaEnergy(self, indexArray):
        return -2*(self.neighborEnergy(indexArray) + self.getIndex(indexArray)*self.fieldB)
    def energyIndex(self,en):
        out = int((en+self.MAX_E)/4)
        ind = self.dimension
        sub = 0
        for i in range(0,self.dimension):
            dim = self.dimension - i
            if(out>=ind):
                sub+=dim-1
            ind+=dim-1
        for i in range(1,self.dimension):
            dim = i
            if(out>self.MAX_E/2-ind):
                sub+=i-1
            ind-=i
        return out - sub;

grd = IsingGrid(2,4)
print(grd)
