import numpy as np
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
from correlator_tools import CorrelatorTools
plt.close('all')


def main():
    corr = CorrelatorTools(a = 0.1141, K = 1000, L = 16)
    # Define the list of flavours to run
    flavours_to_run = ['ll', 'sl', 'ss', 'ls']

    # Lists to store results for all flavours
    all_t = []
    all_db_mean = []
    all_db_err = []
    all_db_samples = []
    for flavour in flavours_to_run:
        print(f"Flavour: {flavour}")
    
        # Empty list for all configurations
        all_disco = []
    
        # First and second loop flavours
        q1 = flavour[0]
        q2 = flavour[1]

        # Sources
        ihits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

        for conf in range(1500, 2661, 40):

            # Empty lists for each config, one loop per source
            loops_q1 = []
            loops_q2 = []

            for ihit in ihits:
                filename1 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc16/Z2Loop_' + q1 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
                t, loop1 = np.loadtxt(filename1, unpack=True)
                loops_q1.append(loop1)

                filename2 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc16/Z2Loop_' + q2 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
                t, loop2 = np.loadtxt(filename2, unpack=True)
                loops_q2.append(loop2)

            
            loops_q1 = np.array(loops_q1)
            loops_q2 = np.array(loops_q2)

            # Append to disco list once per config
            if q1 == q2:
                all_disco.append(corr.disco_correlator(loops_q1))
            else:
                all_disco.append(corr.disco_correlator(loops_q1, loops_q2))

        folded_disco = corr.twoptf_c(all_disco)

        # Mean and error of D(t)
        db_mean, db_samples = corr.bootstrap(folded_disco)
        db_err = corr.bootstrap_std(db_samples)

    #    # Plot of the average disconnected correlator with bootstrap error bars
    #     plt.figure()
    #     plt.yscale('log')
    #     plt.errorbar(t[2:-1], db_mean[2:-1], yerr=db_err[2:-1])
    #     plt.xlabel('t')
    #     plt.ylabel('D_mean')
    #     plt.title('Disconnected correlator ' + flavour + ' against time')
    #     plt.show()

        # plt.figure()
        # plt.plot(t[2:-1], db_mean[2:-1] / db_err[2:-1])
        # plt.xlabel('t')
        # plt.ylabel('D_mean / D_err')
        # plt.title('Relative error')
        # plt.show()
        # Append results to lists
        all_t.append(t)
        all_db_mean.append(db_mean[2:-1])
        all_db_err.append(db_err[2:-1])
        all_db_samples.append(db_samples)
    return all_t, all_db_mean, all_db_err, all_db_samples

if __name__ == "__main__":
    main()