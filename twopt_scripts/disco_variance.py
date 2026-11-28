
import matplotlib
import numpy as np
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

        flavour = 'll'
        q1 = flavour[0]
        timestep = 6

        ihits = list(range(32))

        n_list = [2, 4, 8, 16, 32]

        loops_q1 = []
        for ihit in ihits:
            filename1 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc64/Z2Loop_' + q1 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
            t, loop1 = np.loadtxt(filename1, unpack=True, dtype=complex)
            loops_q1.append(loop1)

        all_loops.append(np.array(loops_q1))

    variances = []

    for n in n_list:
        # Split sources into 2
        half = n // 2

        disco_values = []
        disco_imag_values = []
        loop_values_A = []
        loop_values_B = []

        for loops_q1 in all_loops:
            # Disconnected correlator for this configuration
            values_at_t = loops_q1[:n, timestep]
            A = values_at_t[:half]
            B = values_at_t[half:]

            mean_A = A.sum() / half
            mean_B = B.sum() / half

            # Disconnected correlator
            D_est = mean_A * mean_B / corr.L**3

            disco_values.append(D_est.real)
            disco_imag_values.append(D_est.imag)

            # Store individual loop estimates for vacuum subtraction
            loop_values_A.append(mean_A.real)
            loop_values_B.append(mean_B.real)

        print('Disco mean:', np.mean(disco_values))

        # Imaginary part should be about zero
        imag_mean = np.mean(disco_imag_values)
        imag_std = np.std(disco_imag_values) / np.sqrt(len(disco_imag_values))
        print('Im mean:', imag_mean, '+/-', imag_std)

        # Bootstrap over configurations
        disco_mean, disco_samples = corr.bootstrap(disco_values)

        mean_A, samples_A = corr.bootstrap(loop_values_A)
        mean_B, samples_B = corr.bootstrap(loop_values_B)

        # Subtract vacuum contribution
        disco_samples = disco_samples - samples_A * samples_B / corr.L**6

        variance = (corr.bootstrap_std(disco_samples))**2
        variances.append(variance)

    # Plot variance against number of sources
    n_array = np.array(n_list)
    plt.figure()
    plt.loglog(n_array, variances, 'o-', label='vacuum subtracted')
    plt.loglog(n_array, variances[0] * (2 / n_array), label='1/n')
    plt.loglog(n_array, variances[0] * (2 / n_array)**2, label='1/n^2')
    plt.xlabel('Number of sources, N')
    plt.ylabel('Variance at t = ' + str(timestep))
    plt.legend()
    plt.savefig('Disco_Variance3.png')


if __name__ == "__main__":
    main()
