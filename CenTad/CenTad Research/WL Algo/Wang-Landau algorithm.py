##standard libraries
import numpy as np
import random, sys

##user defined libraries
from WLObjects import IsingGrid, PottsGrid, EnergyHistogram, SpinGlass
from Plot import plotArraytoFile

#Wang-Landau algorithm. Meat of the program...
def wangLandau(grid):
    hist = EnergyHistogram()
    fFactor = np.e
    energy = grid.calcEnergy()
    fSweeps = 0
    while fFactor > np.exp(fExp):
        count = 0
        while True:
            for iterations in range(0,grid.numSites):
                randIndex = [0.0]*grid.dimension
                for i in range(0,grid.dimension):
                    randIndex = energy, grid.GetIndex(randIndex)
                 enew, tempVal = energy, grid.getIndex(randIndex)
                 if((gridMode==0) | (gridMode==2)):
                     enew = energy + grid.deltaEnergy(randIndex)
                elif(gridMode==1):
                    delta = -grid.neighborEnergy(randIndex)
                    tempVal = grid.flipIndex(randIndex)
                    delta += grid.neighborEnergy(randIndex)
                    enew = energy + delta
                gNew, gOld = hist.gValue(enew), hist.gValue(energy)
                gRatioLog = gOld - gNew
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
            count+=1
            #if(isFlatStd(h,np.sqrt(np.log(fFactor))+fExp) & ((count > minIterations) | (0 not in h) ) ):
            if((count%flatnessTestPeriod)==0):
                if(isFlatAverage(hist.h) & (isFlatStd(hist.h,np.sqrt(np.log(fFactor))+fExp)) & ((count > minIterations) | (0 not in hist.h) ) ):
                    minG=min(hist.logG)
    #               for i in range(0,hist.N):
    #                    logG[i]-=minG       #normalize logG vector to prevent overflows of exp()
                    break
                if(verbose):
                    print "Iterations:" + str(fSweeps) + "\tSweeps:" + str(count) + "\tNumber of energy states:" + str(hist.N)
        fFactor=np.sqrt(fFactor)
        fSweeps+=1
        print "F="+str(fFactor)+"\tSweeps:"+str(count)+"\tNumber of Energy States counted:"+str(hist.N)
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
       # pow(out[i],sumLogG)
    return [hist.logG, out, hist.energies]
