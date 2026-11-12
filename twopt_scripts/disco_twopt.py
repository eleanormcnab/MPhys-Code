import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
from correlator_tools import CorrelatorTools
plt.close('all')

def main():
    corr = CorrelatorTools(a = 0.11, K = 1000, L = 16)

    # Empty list for all configurations
    all_disco = []
    # ll: light-light
    flavour1 = 'll'
    # ss: strange-strange
    flavour2 = 'ss'
    # ls: light, strange
    flavour3 = 'ls'
    #sl: strange, light
    flavour4 = 'sl'
    # Pick which one to run
    flavour = flavour3
    # First and second loop flavours
    q1 = flavour[0]
    q2 = flavour[1]

    # Noise hits
    ihits = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    for conf in range(1500, 2661, 40):

        # Empty lists for each config, one loop per noise hit
        loops_q1 = []
        loops_q2 = []

        for ihit in ihits:
            filename1 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc16/Z2Loop_' + q1 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
            t, loop1 = np.loadtxt(filename1, unpack=True)
            loops_q1.append(loop1)

            filename2 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc16/Z2Loop_' + q2 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
            t, loop2 = np.loadtxt(filename2, unpack=True)
            loops_q2.append(loop2)

        # Arrays of shape (N, T)
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
    db_average = np.mean(db_mean)
    print(db_mean)
    print(db_err)

    # Plot of the average disconnected correlator with bootstrap error bars
    plt.figure()
    plt.errorbar(t[2:-1], db_mean[2:-1], yerr=db_err[2:-1])
    plt.xlabel('t')
    plt.ylabel('D_mean')
    plt.title('Disconnected correlator ' + flavour + ' against time')
    plt.savefig("Disconnected_ls.png")

    plt.figure()
    plt.plot(t[2:-1], db_mean[2:-1] / db_err[2:-1])
    plt.xlabel('t')
    plt.ylabel('D_mean / D_err')
    plt.title('Relative error')
    plt.savefig("Relative_err_ls_disconnected.png")


if __name__ == "__main__":
    main()