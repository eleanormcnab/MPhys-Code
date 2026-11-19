import numpy as np
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
import warnings
warnings.filterwarnings('ignore')

from correlator_tools import CorrelatorTools

def main():
    corr = CorrelatorTools(a = 0.1141, K = 1000, L = 16)

    # Define the list of flavours to run
    flavours_to_run = ['ll', 'sl', 'ss']

    # Lists to store results for all flavours
    all_t = []
    all_cb_mean = []
    all_cb_err = []
    all_masses_n = []
    all_masses_err_n = []
    all_masses = []
    all_masses_err = []
    all_cb_samples = []

    for flavour in flavours_to_run:
        print(f"Finding mass for: {flavour}")

        # Empty lists for all configurations
        all_twopt = []
        
        # Source 1: Point sources
        source1 = 'pt'
        # Source 2: Z2 sources
        source2 = 'Z2'

        # Pick which ones to run
        source = source1 

        for conf in range(1500, 2661, 40):

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
        cb_err = corr.bootstrap_std(cb_samples)
        
        # Numerical method for mass with bootstrap error
        meff_samples_n = np.array([corr.num_meff(c) for c in cb_samples])
        meff_b_n = meff_samples_n.mean(axis=0)
        meff_b_err_n = corr.bootstrap_std(meff_samples_n)

        # Physical mass and error
        mass_MeV_b_n = corr.physical_mass(meff_b_n, 12, 20)
        mass_samples_n = np.array([corr.physical_mass(m, 12, 20) for m in meff_samples_n])
        mass_MeV_b_err_n = corr.bootstrap_std(mass_samples_n)

        # Identify plateau from numerical method (t=12 to 20)
        # Use this range to define where to fit the correlator
        t_fit = t[12:20]
        
        mass_samples_f = []
        T = len(folded_twopt[0])
        for c in cb_samples:
            c_fit = c[12:20]
            m_fit_lattice = corr.correlators_fit(c_fit, t_fit, T)
            mass_samples_f.append(m_fit_lattice * corr.hbarc / corr.a)

        # Physical mass and error      
        mass_samples_f = np.array(mass_samples_f)
        mass_MeV_b_f = np.nanmean(mass_samples_f)
        mass_MeV_b_err_f = corr.bootstrap_std(mass_samples_f)

        # Print effective mass
        print(f"{flavour} Numerical: {mass_MeV_b_n} +/- {mass_MeV_b_err_n}")
        print(f"{flavour} Fit:       {mass_MeV_b_f} +/- {mass_MeV_b_err_f}")

        # Append results to lists
        all_t.append(t[2:-1])
        all_cb_mean.append(cb_mean[2:-1])
        all_cb_err.append(cb_err[2:-1])
        all_cb_samples.append(cb_samples[:, 2:-1])
        all_masses_n.append(mass_MeV_b_n)
        all_masses_err_n.append(mass_MeV_b_err_n)
        all_masses.append(mass_MeV_b_f)
        all_masses_err.append(mass_MeV_b_err_f)

    return all_t, all_cb_mean, all_cb_err, all_masses_n, all_masses_err_n, all_masses, all_masses_err, all_cb_samples

if __name__ == "__main__":
    main()