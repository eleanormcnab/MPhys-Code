import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from correlator_tools import CorrelatorTools
import disco_twopt as disconnected
import twopt_adapted as connected
plt.close('all')

def main():
    corr = CorrelatorTools(a = 0.1141, K = 1000, L = 16)

    # Connected and disconnected results
    all_t_d, all_db_mean, all_db_err, all_db_samples = disconnected.main()
    all_t, all_cb_mean, all_cb_err, all_masses_n, all_masses_err_n, all_masses, all_masses_err, all_cb_samples = connected.main()

    t = all_t[0]

    # Elements M11 and M22 of quark flavour basis matrix
    M11_qf_samples = all_cb_samples[0] - 2.0 * all_db_samples[0]
    M22_qf_samples = all_cb_samples[2] - 2.0 * all_db_samples[2]

    # Pion twopoint function
    pion_qf_samples = all_cb_samples[0]

    # Mean and bootstrap error
    mean_ll = np.mean(M11_qf_samples, axis=0)
    err_ll = corr.bootstrap_std(M11_qf_samples)
    mean_ss = np.mean(M22_qf_samples, axis=0)
    err_ss = corr.bootstrap_std(M22_qf_samples)
    mean_pion = np.mean(pion_qf_samples, axis=0)
    err_pion = corr.bootstrap_std(pion_qf_samples)

    #Combined effective masses
    meff_ll_samples = np.array([corr.num_meff(sample) for sample in M11_qf_samples])
    mean_meff_ll = np.nanmean(meff_ll_samples, axis=0)
    err_meff_ll = corr.bootstrap_std(meff_ll_samples)

    meff_ss_samples = np.array([corr.num_meff(sample) for sample in M22_qf_samples])
    mean_meff_ss = np.nanmean(meff_ss_samples, axis=0)
    err_meff_ss = corr.bootstrap_std(meff_ss_samples)

    meff_pion_samples = np.array([corr.num_meff(sample) for sample in pion_qf_samples])
    mean_meff_pion = np.nanmean(meff_pion_samples, axis=0)
    err_meff_pion = corr.bootstrap_std(meff_pion_samples)


    
    # Plots of effective masses
    plt.figure()
    plt.errorbar(t, mean_meff_ll, yerr=err_meff_ll, label='M11 (ll)')
    plt.errorbar(t, mean_meff_ss, yerr=err_meff_ss, label='M22 (ss)')
    plt.errorbar(t, mean_meff_pion, yerr=err_meff_pion, label='Pion')
    plt.xlabel('t')
    plt.ylabel('Effective Mass')
    plt.title('Effective Masses')
    plt.legend()
    plt.show()

    # Plotting twopoint functions
    plt.figure()
    plt.errorbar(t, mean_ll, yerr=err_ll, label='M11 (ll)')
    plt.errorbar(t, mean_ss, yerr=err_ss, label='M22 (ss)')
    plt.errorbar(t, mean_pion, yerr=err_pion, label='Pion')
    plt.xlabel('t')
    plt.ylabel('Correlators')
    plt.title('Combined Correlators')
    plt.legend()
    plt.show()

    # Arrays start at t = 2, so t = 12 to 20 is indices 10 to 18
    print(f"Plateau (ll) = {np.mean(mean_ll[10:19])}")
    print(f"Plateau (ss) = {np.mean(mean_ss[10:19])}")

if __name__ == "__main__":
    main()