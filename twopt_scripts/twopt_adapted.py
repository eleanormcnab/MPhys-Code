import numpy as np
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
from correlator_tools import CorrelatorTools
plt.close('all')

def main():
    corr = CorrelatorTools(a = 0.11, K = 1000)

    # Empty lists for all configurations
    all_twopt = []
    # ll: pions
    flavour1 = 'll'
    # sl: kaons
    flavour2 = 'sl'
    # Source 1: Point sources
    source1 = 'pt'
    # Source 2: Z2 sources
    source2 = 'Z2'

    # Pick which ones to run
    flavour = flavour1
    source = source1

    for conf in range(1500, 1781, 40):
        # Empty list for each config
        c_this_config = []

        # Source times
        for ts in [0, 4, 8, 12, 16, 20, 24, 28]:
            filename = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_64/' + source + str(ts) + '_' + flavour + '.' + str(conf) + '.dat'
            t, twopt = np.loadtxt(filename, unpack=True)

            # Only the Z2 files are divided by the lattice volume
            if source == source2:
                twopt = corr.z_files(twopt)
            
            c_this_config.append(twopt)

        # Append to twopt list once per config, after all the source times are loaded
        all_twopt.append(corr.source_average(c_this_config, [0, 4, 8, 12, 16, 20, 24, 28]))
    folded_twopt = corr.twoptf_c(all_twopt)
   

    # Mean and error of C(t)
    cb_mean, cb_samples = corr.bootstrap(folded_twopt)

    # Effective mass of every bootstrap sample
    meff_samples = np.array([corr.meff(c) for c in cb_samples])
    # Effective masses using both jackknife and bootstrap methods, and bootstrap errors on each m_eff
    meff_b = meff_samples.mean(axis=0)
    meff_b_err = corr.bootstrap_std(meff_samples)
    print("Meff calculated")
   # Plots of the average two-point function
    plt.figure()
    plt.semilogy(t[2:-1], cb_mean[2:-1])
    plt.xlabel('t')
    plt.ylabel('C_mean')
    plt.title('C_mean against time')
    plt.show()
    plt.clf() 
    # Plot of the effective mass with bootstrap error bars
    # Start at t = 2 so the plot is symmetric
    plt.errorbar(t[2:-1], meff_b[2:-1], yerr=meff_b_err[2:-1])
    plt.xlabel('t')
    plt.ylabel('m_eff')
    plt.title("Bootstrap effective mass against time")
    plt.show()


    # Mass in MeV for with bootstrap error
    mass_MeV_b = corr.physical_mass(meff_b, 12, 16)
    mass_samples = np.array([corr.physical_mass(m, 12, 16) for m in meff_samples])
    mass_MeV_b_err = corr.bootstrap_std(mass_samples)
    
    # Print bootstrap method mass and error
    print(f"Mass from bootstrap method = {mass_MeV_b} +/- {mass_MeV_b_err} MeV")


if __name__ == "__main__":
    main()