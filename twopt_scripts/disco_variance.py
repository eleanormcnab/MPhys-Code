import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from correlator_tools import CorrelatorTools
plt.close('all')


def main():
    """Finding variance of one timestep for one configuration, plotting it against number of sources, to see if number sources decreases variance (should go as 1/N^2)"""
    corr = CorrelatorTools(a=0.1141, K=1000, L=16)

    # One configuration, ll flavour
    conf = 1500
    flavour = 'll'
    q1 = flavour[0]
    # #Pick one timestep
    timestep = 16
    # Stochastic sources list
    ihits = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
             27, 28, 29, 30, 31]
    # 2 sources, 4 sources, and so on. 
    n_list = [2, 4, 8, 16]


    # Load loop data for one configuration
    loops_q1 = []
    for ihit in ihits:
        filename1 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc16/Z2Loop_' + q1 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
        t, loop1 = np.loadtxt(filename1, unpack=True)
        loops_q1.append(loop1)
    loops_q1 = np.array(loops_q1)

    # Empty variance list
    variances = []

    # Calculate variance for each number of sources
    for n in n_list:

        disco_values = []
        # Easier to use 32 - n to select each datapoint
        for i in range(32 - n):
            # D(t) from n sources, starting at source i
            print(i)
            D = corr.disco_correlator(loops_q1[i : i+n])

            # Keep only the value at the chosen timestep
            disco_values.append(D[timestep])

        variance = np.var(disco_values)
        variances.append(variance)

        print(f'n = {n}: variance = {variance}')
    # Plot variance against number of sources
    n_array = np.array(n_list)
    print(variances[0])
    plt.figure()
    plt.loglog(n_array, variances, 'o-', label='data')
    # Comparing this with 1/n^2 (2 in denominator for first number of sources)
    plt.loglog(n_array, variances[0] * (2 / n_array)**2, label='1/n^2')
    plt.xlabel('Number of sources, N')
    plt.ylabel('Variance at t = ' + str(timestep))
    plt.legend()
    plt.savefig('Disco_variance.png')


if __name__ == "__main__":
    main()