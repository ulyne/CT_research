import numpy as np
import random
import matplotlib.pyplot as plt

#initial array
def init_spin_array(row, col):
    return np.ones((row, col))

#nearest neighbours
def neighbors(spin_array, lattice, x, y):
    left   = (x, y - 1)
    right  = (x, (y + 1) % lattice)
    top    = (x - 1, y)
    bottom = ((x + 1) % lattice, y)

    return [spin_array[left[0], left[1]],
            spin_array[right[0], right[1]],
            spin_array[top[0], top[1]],
            spin_array[bottom[0], bottom[1]]]

#energy of the configuration
def energy(spin_array, lattice, x ,y):
    return 2 * spin_array[x, y] * sum(neighbors(spin_array, lattice, x, y))



#main code
def main():
    #defining the number of initial sweeps, the lattice size, and number of monte carlo sweeps
    RELAX_SWEEPS = 50
    lattice = 4
    sweeps = 1000

    for temperature in np.arange(0.1, 4.0, 0.2):
        e1= e0 = 0
        #setting up initial variables
        spin_array = init_spin_array(lattice, lattice)
        mag = np.zeros(sweeps + RELAX_SWEEPS)
        spec = np.zeros(sweeps + RELAX_SWEEPS)
        Energy = np.zeros(sweeps + RELAX_SWEEPS)
        # the Monte Carlo
        for sweep in range(sweeps + RELAX_SWEEPS):
            for i in range(lattice):
                for j in range(lattice):
                    e = energy(spin_array, lattice, i, j)
                    if e <= 0:
                        spin_array[i, j] *= -1
                    elif np.exp((-1.0 * e)/temperature) > random.random():
                        spin_array[i, j] *= -1

            #Thermodynamic Variables 

            #Magnetization
            mag[sweep] = abs(sum(sum(spin_array))) / (lattice ** 2)

            

            #Energy
            Energy[sweep] = energy(spin_array,lattice,i,j)/ (lattice ** 2)


            #Specific Heat
            e0 = e0 + energy(spin_array,lattice,i,j)               
            e1 = e1 + energy(spin_array,lattice,i,j) *energy(spin_array,lattice,i,j)
            
        e1 = e1/sweeps
        e0 = e0/sweeps
 
        spec[sweep]=((e1/lattice - e0*e0/(lattice*lattice)) / (temperature * temperature))


        #Printing the thermodynamic variables    
    
        print(temperature,sum(Energy[RELAX_SWEEPS:]) / sweeps, sum(mag[RELAX_SWEEPS:]) / sweeps, spec[RELAX_SWEEPS:] / sweeps)
        


main()