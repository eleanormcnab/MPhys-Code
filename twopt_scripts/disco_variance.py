import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from correlator_tools import CorrelatorTools
plt.close('all')


def main():
    """Finding variance of one timestep for all configurations, plotting it against number of sources."""
    corr = CorrelatorTools(a=0.1141, K=1000, L=16)

    # Loop data for every configuration
    all_loops = []

    for conf in range(1500, 2661, 40):
        if conf == 1820:
            continue

        flavour = 'll'
        q1 = flavour[0]
        timestep = 6

        ihits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                 27, 28, 29, 30, 31]

        n_list = [2, 4, 8, 16, 31]

        loops_q1 = []
        for ihit in ihits:
            filename1 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc16/Z2Loop_' + q1 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
            t, loop1 = np.loadtxt(filename1, unpack=True)
            loops_q1.append(loop1)

        all_loops.append(np.array(loops_q1))

    variances = []

    for n in n_list:
        # Split sources into 2 
        half = n // 2

        disco_values = []
        loop_values = []

        for loops_q1 in all_loops:
            # Disconnected correlator for this configuration, but splitting sources
            values_at_t = loops_q1[:n, timestep]
            A = values_at_t[:half]
            B = values_at_t[half:]
            mean_A = A.sum() / half
            mean_B = B.sum() / half
            # Normalise by volume and multiply separate sets to remove bias
            D_est = (mean_A) * (mean_B) / corr.L**3
            disco_values.append(D_est)

            # Average loop for this configuration
            loops_avg_time = np.mean(loops_q1[:n], axis=1)
            A_v = loops_avg_time[:half]
            B_v = loops_avg_time[half:]
            mean_Av = A_v.sum() / half
            mean_Bv = B_v.sum() / half
            # Normalise by volume and multiply separate sets
            loop = (mean_Av) * (mean_Bv)  / corr.L**3
            loop_values.append(loop)
            
        print('Disco mean:', np.mean(disco_values))
        print('Loop mean:', np.mean(loop_values))
        # Bootstrap both over configurations
        disco_mean, disco_samples = corr.bootstrap(disco_values)
        loop_mean, loop_samples = corr.bootstrap(loop_values)

        # Subtract vacuum contribution
        disco_samples = disco_samples - loop_samples

        variance = (corr.bootstrap_std(disco_samples))**2
        variances.append(variance)

        print('n = ' + str(n) + ': variance = ' + str(variance))
        print('Disco mean:', np.mean(disco_values))

    # Plot variance against number of sources
    n_array = np.array(n_list)
    plt.figure()
    plt.loglog(n_array, variances, 'o-', label='vacuum subtracted')
    plt.loglog(n_array, variances[0] * (2 / n_array)**2, label='1/n^2')
    plt.xlabel('Number of sources, N')
    plt.ylabel('Variance at t = ' + str(timestep))
    plt.legend()
    plt.savefig('Disco_variance.png')


if __name__ == "__main__":
    main()