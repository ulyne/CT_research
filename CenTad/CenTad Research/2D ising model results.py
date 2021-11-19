import matplotlib.pyplot as plt

x1 = [0.1, 0.30000000000000004, 0.5000000000000001, 0.7000000000000001, 0.9000000000000001, 1.1000000000000003, 1.3000000000000003, 1.5000000000000004, 1.7000000000000004, 1.9000000000000004, 2.1000000000000005, 2.3000000000000007, 2.5000000000000004, 2.7000000000000006, 2.900000000000001, 3.1000000000000005, 3.3000000000000007, 3.500000000000001, 3.7000000000000006, 3.900000000000001]

y1 = [ 0.16175, 0.176, 0.222, 0.227,  0.20025,  0.27475, 0.314, 0.33775, 0.377, 0.4225, 0.448, 0.474, 0.48725, 0.49475, 0.496, 0.499, 0.5, 0.5, 0.5, 0.5] 

y2 = [1.0, 1.0, 1.0, 1.0, 0.9995, 0.99875, 0.996125, 0.9885, 0.973375, 0.939125, 0.90225, 0.828875, 0.74275, 0.68875, 0.654625, 0.554, 0.53775, 0.495, 0.46025, 0.4365]
       
y3 = [6.278999999999999,
 0.6983152843333331,
 0.25139373584470226,
 0.12826211024397297,
 0.07753658916519668,
 0.05194081836678378,
 0.036798264665719935,
 0.027439527634392794,
 0.020890540560197554,
 0.016195816182427183,
 0.012357296370863313,
 0.009433001142870061,
 0.007206427545667451,
 0.005846330452010217,
 0.004518479749870508,
 0.0035921590336719525,
 0.002870465065453035,
 0.002419031292789248,
 0.002005607134198224,
 0.001609394223035462
]

plt.plot(x1, y1, label = "energy")
plt.xlabel('temperature')
plt.ylabel('energy')
plt.title('2D Ising Model of 4x4 lattice structure\n energy against temperature')
plt.show()

plt.plot(x1, y2, label = 'magnetism')
plt.xlabel('temperature')
plt.ylabel('magnetism')
plt.title('2D Ising Model of 4x4 lattice structure\n magnetism against temperature')
plt.show()

plt.plot(x1, y3, label = 'specific heat')
plt.xlabel('temperature')
plt.ylabel('specific heat')
plt.title('2D Ising Model of 4x4 lattice structure\n specific heat against temperature')
plt.show()