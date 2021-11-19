import random
import numpy as np
import bisect

class EnergyHistogram:
    def __init__(self):
        self.N=0
        self.h = []
        self.logG = []
        self.energies = []
        self.energyTolerance = 0.0000001
    def addValue(self,energy, fFactor):
        if(self.N==0):
            self.N=1
            self.h.append(1)
            self.energies.append(energy)
            self.logG.append(fFactor)
            return
        index = bisect.bisect_left(self.energies, energy)
        if((index<self.N) & (index >= 0)):
            if((np.abs(self.energies[index] - energy) < self.energyTolerance)):
                self.logG[index]+=fFactor
                self.h[index]+=1
                return
            if(index > 0):
                if((np.abs(self.energies[index - 1] - energy) < self.energyTolerance)):
                    self.logG[index - 1]+=fFactor
                    self.h[index - 1]+=1
                    return
        self.logG = self.logG[0:index] + [fFactor] + self.logG[index:self.N]
        self.energies = self.energies[0:index] + [energy] + self.energies[index:self.N]
        self.h = self.h[0:index] + [1] + self.h[index:self.N]
        self.N+=1
    def gValue(self,energy):
        for i in range(0,self.N):
            if(np.abs(energy - self.energies[i]) < self.energyTolerance):
                return self.logG[i]
        return 0.0
    def resetHistogram(self):
        self.h = [0]*self.N

def plotToFile(plotFunction,plotRange,fileOut,header='',log10plot=False,verbose=False):
    from subprocess import call
    file = open(fileOut, 'w')
    
    file.write("#" + header)
    plotVals = np.linspace( plotRange[0],plotRange[1], plotRange[2] )
    for k in range(0,len(plotVals)):
        x = plotVals[k]
        if(log10plot):
            x=10**plotVals[k]
        out=plotFunction(x)
        file.write( str(x) )
        if( isinstance(out,list) ):
            for i in range(0,len(out)):
                file.write('\t' + str(out[i]) )
        else:
            file.write('\t' + str(out) )
        file.write('\n')
        if(verbose):
            print (str(float(k)*100.0/len(plotVals)) + " % done")
    file.close()

#used for plotting a matrix of data points (to a file for gnuplot)
def plotArrayToFile(plotData,fileOut,header='',writeIndex=True):
    from subprocess import call
    file = open(fileOut, 'w')
    
    file.write("#" + header + '\n')
    for k in range(0,len(plotData)):
        if(writeIndex):
            file.write(str(k) + '\t')
        if( isinstance(plotData[k],list)):
            for i in range(0,len(plotData[k])):
                if(i!=0):
                    file.write('\t')
                file.write(str(plotData[k][i]) + '\t')
        else:
            file.write(str(plotData[k]))
        file.write('\n')
    file.close()
    
#averages multiple runs of a function with the same arguments (good for monte carlo functions). Also produces standard deviations sqrt(<X^2> - <x>^2)
###Now compatible with functions that give arrays as output. Averages each element & produces deviations
def averageFunc(func,numAverages,funcArg):
    f=func(funcArg)
    if(not isinstance(f,list) ):
        sum, sumSqr = 0.0, 0.0
        for i in range(0,numAverages):
            sum += f
            sumSqr += f**2
            f=func(funcArg)
        return [sum/numAverages,np.sqrt(sumSqr/numAverages-(sum/numAverages)**2)]
    else:
        dim=len(f)
        sum, sumSqr = [0.0]*dim, [0.0]*dim
        for i in range(0,numAverages):
            for j in range(0,dim):
                sum[j] += f[j]
                sumSqr[j] += f[j]**2
            f=func(funcArg)
        out = []
        for j in range(0,dim):
            out.append(sum[j]/numAverages)
            out.append(np.sqrt(sumSqr[j]/numAverages - (sum[j]/numAverages)**2))
        return out

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

class SpinGlass:
    def __init__(self,dimension,N):
        self.dimension = dimension
        self.gridSize = N
        self.numSites = N**dimension
        self.MAX_E = dimension*N**dimension
        self.array = [0]*N**dimension
        self.j = [0]*N**dimension
        self.fieldB = 0.0
        for i in range(0,N**dimension):
            self.array[i] = -1+2*int(random.random()*2)
            self.j[i] = -1+2*int(random.random()*2)
    
    def getIndex(self,indexArray):
        index = 0;
        for i in range(0,self.dimension):
            index+=self.gridSize**(self.dimension - i - 1)*indexArray[self.dimension-i-1]
        return self.array[index]
    def getJ(self,indexArray):
        index = 0;
        for i in range(0,self.dimension):
            index+=self.gridSize**(self.dimension - i - 1)*indexArray[self.dimension-i-1]
        return self.j[index]
    def flipIndex(self,indexArray):
        index = 0;
        for i in range(0,self.dimension):
            index+=self.gridSize**(self.dimension - i - 1)*indexArray[self.dimension-i-1]
        self.array[index]*=-1
    def neighborHalfEnergy(self,indexArray):
        sum=0.0
        val=self.getIndex(indexArray)
        jVal=self.getJ(indexArray)
        dim = self.dimension
        N = self.gridSize
        for i in range(0,dim):
            ind = indexArray[0:i] + [ (indexArray[i] + 1) % N ] + indexArray[i+1:dim] #sum up each neigbor by adding one to each coordinate dimension
            sum+= -self.getIndex(ind)*val*self.getJ(ind)*jVal
        return sum
    def neighborEnergy(self,indexArray):
        sum=0.0
        val=self.getIndex(indexArray)
        jVal=self.getJ(indexArray)
        dim = self.dimension
        N = self.gridSize
        for i in range(0,dim):
            ind1 = indexArray[0:i] + [ (indexArray[i] + 1) % N ] + indexArray[i+1:dim] #sum up each neigbor by adding one and subtracting one to each coordinate dimension
            ind2 = indexArray[0:i] + [ (indexArray[i] - 1) % N ] + indexArray[i+1:dim]
            sum+= -self.getIndex(ind1)*val*jVal*self.getJ(ind1)
            sum+= -self.getIndex(ind2)*val*jVal*self.getJ(ind2)
        return sum
    def calcEnergy(self,indexArrayPartial = []):
        if(len(indexArrayPartial)==self.dimension):
            return self.neighborHalfEnergy(indexArrayPartial)
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
        return out- sub;

dim = 2
periodicBoundaries=True
graphProgress = True
sizeN=4
fExp=0.1**7
minIterations = 30
gridMode= 0
verbose=False
flatnessTestPeriod = 1000

fileOut = 'output.txt'

if(graphProgress):
    import matplotlib; matplotlib.use('TKAgg'); import matplotlib.pyplot as plt
    plt.ion()

#Wang-Landau algorithm. Meat of the program...
def wangLandau(grid):
    hist = EnergyHistogram()
    fFactor = np.e
    energy = grid.calcEnergy()
    fSweeps=0
    while fFactor > np.exp(fExp):
        hist.resetHistogram()
        count = 0
        while True:
            for iterations in range(0,grid.numSites):
                randIndex=[0.0]*grid.dimension
                for i in range(0,grid.dimension):
                    randIndex[i]=int(random.random()*grid.gridSize)
                enew, tempVal = energy, grid.getIndex(randIndex)
                if((gridMode==0) | (gridMode==2)):
                    enew = energy + grid.deltaEnergy(randIndex)
                elif(gridMode==1):
                    delta = -grid.neighborEnergy(randIndex)
                    tempVal = grid.flipIndex(randIndex)
                    delta += grid.neighborEnergy(randIndex)
                    enew = energy + delta
                gNew, gOld = hist.gValue(enew), hist.gValue(energy)
                gRatioLog = gOld- gNew
                if( (gNew < gOld) | (np.log(random.random()) < (gRatioLog)) ):
                    hist.addValue(enew,fFactor)
                    if((gridMode==0) | (gridMode==2)):
                        grid.flipIndex(randIndex)
                    energy = enew
                else:
                    hist.addValue(energy,fFactor)
                    if(gridMode==1):
                        grid.setIndex(randIndex,tempVal)

            if(graphProgress):
                plt.figure(1)
                plt.cla()
                plt.ylabel('Entropy (Log[state Density])')
                plt.plot(hist.energies,hist.logG)
                plt.figure(2)
                plt.cla()
                plt.ylabel('Current Histogram')
                plt.plot(hist.h)
                plt.draw()
                plt.show()
            count+=1
            #if(isFlatStd(h,np.sqrt(np.log(fFactor))+fExp) & ((count > minIterations) | (0 not in h) ) ):
            if((count%flatnessTestPeriod)==0):
                if(isFlatAverage(hist.h) & (isFlatStd(hist.h,np.sqrt(np.log(fFactor))+fExp)) & ((count > minIterations) | (0 not in hist.h) ) ):
                    minG=min(hist.logG)
    #               for i in range(0,hist.N):
    #                    logG[i]-=minG       #normalize logG vector to prevent overflows of exp()
                    break
                if(verbose):
                    print ("Iterations:" + str(fSweeps) + "\tSweeps:" + str(count) + "\tNumber of energy states:" + str(hist.N))
        fFactor=np.sqrt(fFactor)
        fSweeps+=1
        print ("F="+str(fFactor)+"\tSweeps:"+str(count)+"\tNumber of Energy States counted:"+str(hist.N))
    out = [0.0]*hist.N
    baseline = min(hist.logG)
    for i in range(0,hist.N):
        hist.logG[i]-=baseline
        out[i]=np.exp(hist.logG[i])
        hist.energies[i]/=grid.numSites
    #normConst = pow(float(dim**N)/sum(out),1.0/sumLogG)
    normConst = float(dim**grid.gridSize)/sum(out)
    normConstLog = np.log(normConst)
    #for i in range(0,len(logG)):
        #out[i]*=normConst
        #logG[i]+=normConstLog
        #pow(out[i],sum.logG)
    return [hist.logG, out, hist.energies]

def isFlatAverage(histogram):
    avg = np.mean(histogram)
    for i in range(0,len(histogram)):
        if(histogram[i]<0.8*avg):
            return False
    return True
def isFlatStd(histogram, tolerance):
    avg = np.mean(histogram)
    std = np.std(histogram)
    return (std < (avg*tolerance))

def closeGraphs():
    plt.figure(1)
    plt.close()
    plt.figure(2)
    plt.close()

#main loop of program

grd = IsingGrid(dim,sizeN)

output = wangLandau(grd)

plotArrayToFile(output[0],fileOut)
