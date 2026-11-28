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
        # Checking if imaginary parts are zero
        disco_imag_values = []
        loop_values = []

        for loops_q1 in all_loops:
            # Disconnected correlator for this configuration, but splitting sources
            values_at_t = loops_q1[:n, timestep]
            A = values_at_t[:half]
            B = values_at_t[half:]
            mean_A = A.sum() / half
            mean_B = B.sum() / half
            # Normalise by volume and multiply separate sets to remove bias
            # D_est.real = Re[A]Re[B] - Im[A]Im[B]
            # D_est.imag = Re[A]Im[B] + Im[A]Re[B] - should be zero
            D_est = (mean_A) * (mean_B) / corr.L**3
            disco_values.append(D_est.real)          
            disco_imag_values.append(D_est.imag)     

            # Average loop for this configuration
            loops_avg_time = np.mean(loops_q1[:n], axis=1)
            A_v = loops_avg_time[:half]
            B_v = loops_avg_time[half:]
            mean_Av = A_v.sum() / half
            mean_Bv = B_v.sum() / half
            # Normalise by volume and multiply separate sets
            loop = (mean_Av) * (mean_Bv) / corr.L**3
            loop_values.append(loop.real)            
            
        print('Disco mean:', np.mean(disco_values))
        print('Loop mean:', np.mean(loop_values))
        
        # Imaginary part should be about zero
        imag_mean = np.mean(disco_imag_values)
        imag_std = np.std(disco_imag_values) / np.sqrt(len(disco_imag_values))
        print('Im  mean:', imag_mean, '+/-', imag_std)

        # Bootstrap both over configurations
        disco_mean, disco_samples = corr.bootstrap(disco_values)
        loop_mean, loop_samples = corr.bootstrap(loop_values)
     
        # Subtract vacuum contribution
        disco_samples = disco_samples - loop_samples

        variance = (corr.bootstrap_std(disco_samples))**2
        variances.append(variance)
    # # Plot variance against number of sources
    # n_array = np.array(n_list)
    # plt.figure()
    # plt.loglog(n_array, variances, 'o-', label='vacuum subtracted')
    # plt.loglog(n_array, variances[0] * (2 / n_array), label='1/n')
    # plt.loglog(n_array, variances[0] * (2 / n_array)**2, label='1/n^2')
    # plt.xlabel('Number of sources, N')
    # plt.ylabel('Variance at t = ' + str(timestep))
    # plt.legend()
    # plt.savefig('Disco_Variance')


if __name__ == "__main__":
    main()
