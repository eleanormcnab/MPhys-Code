import numpy as np
import matplotlib
matplotlib.use('Agg')
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

    t = np.arange(1, 31)

    # Elements M11 and M22 of quark flavour basis matrix
    M11_qf_samples =  all_cb_samples[0] - 2.0 * all_db_samples[0]
    M22_qf_samples = all_cb_samples[2] - all_db_samples[2]

    # Pion twopoint function in quark flavour basis
    pion_qf_samples = all_cb_samples[0]

    # Mean and bootstrap error
    mean_ll = np.mean(M11_qf_samples, axis=0)
    err_ll = corr.bootstrap_std(M11_qf_samples)
    mean_ss = np.mean(M22_qf_samples, axis=0)
    err_ss = corr.bootstrap_std(M22_qf_samples)
    mean_pion = np.mean(pion_qf_samples, axis=0)
    err_pion = corr.bootstrap_std(pion_qf_samples)

    #Combined effective masses, combined bootstrap errors
    meff_ll_samples = np.array([corr.num_meff(sample) for sample in M11_qf_samples])
    mean_meff_ll = np.nanmean(meff_ll_samples, axis=0)
    err_meff_ll = corr.bootstrap_std(meff_ll_samples)

    meff_ss_samples = np.array([corr.num_meff(sample) for sample in M22_qf_samples])
    mean_meff_ss = np.nanmean(meff_ss_samples, axis=0)
    err_meff_ss = corr.bootstrap_std(meff_ss_samples)

    meff_pion_samples = np.array([corr.num_meff(sample) for sample in pion_qf_samples])
    mean_meff_pion = np.nanmean(meff_pion_samples, axis=0)
    err_meff_pion = corr.bootstrap_std(meff_pion_samples)

    # Element M11 and M22 of octet-singlet basis matrix (excluding a constant factor)
    M11_os_samples = 4.0 * all_db_samples[0] - 2.0 * all_cb_samples[0] - 2.0 * all_db_samples[3] + 4.0 * all_db_samples[1] - 2.0 * all_db_samples[2] + 2.0 * all_cb_samples[2]
    mean_M11_os = np.mean(M11_os_samples, axis=0)
    err_M11_os = corr.bootstrap_std(M11_os_samples)
    plateau_M11_os = np.mean(mean_M11_os[12:20])
    print(plateau_M11_os)

    M22_os_samples = 4.0 * all_db_samples[0] - 2.0 * all_cb_samples[0] + 2.0 * all_db_samples[3] - 4.0 * all_db_samples[1] - 2.0 * all_db_samples[2] + 2.0 * all_cb_samples[2]
    mean_M22_os = np.mean(M22_os_samples, axis=0)
    err_M22_os = corr.bootstrap_std(M22_os_samples)
    plateau_M22_os = np.mean(mean_M22_os[12:20])
    print(plateau_M22_os)

    # Combined effective masses, combined bootstrap errors
    meff_M11_os_samples = np.array([corr.num_meff(sample) for sample in M11_os_samples])
    mean_meff_M11_os = np.nanmean(meff_M11_os_samples, axis=0)
    err_meff_M11_os = corr.bootstrap_std(meff_M11_os_samples)

    # Combined effective masses, combined bootstrap errors
    meff_M22_os_samples = np.array([corr.num_meff(sample) for sample in M22_os_samples])
    mean_meff_M22_os = np.nanmean(meff_M22_os_samples, axis=0)
    err_meff_M22_os = corr.bootstrap_std(meff_M22_os_samples)

    
    # # Plots of effective masses
    # plt.figure()
    # plt.errorbar(t, mean_meff_ll[1:-1], yerr=err_meff_ll[1:-1], label='M11 (ll)')
    # plt.errorbar(t, mean_meff_pion[1:-1], yerr=err_meff_pion[1:-1], label='Pion')
    # plt.xlabel('t')
    # plt.ylabel('Effective Mass of Pi0 against M11 of quark flavour basis matrix')
    # plt.title('Effective Masses')
    # plt.legend()
    # plt.savefig('Pi0M11_meff.png')

    # plt.figure()
    # plt.errorbar(t, mean_meff_M11_os[1:-1], yerr=err_meff_M11_os[1:-1], label='M11')
    # plt.errorbar(t, mean_meff_M22_os[1:-1], yerr=err_meff_M22_os[1:-1], label='m22')
    # plt.xlabel('t')
    # plt.ylabel('Effective Mass of M11 and M22 of octet-singlet basis matrix')
    # plt.title('Effective Masses')
    # plt.legend()
    # plt.savefig('M11M22os_meff.png')
    
    # # Plotting twopoint functions
    # plt.figure()
    # plt.errorbar(t, mean_ll[1:], yerr=err_ll[1:], label='M11 (ll)')
    # plt.errorbar(t, mean_pion[1:], yerr=err_pion[1:], label='Pion')
    # plt.xlabel('t')
    # plt.ylabel('Correlators')
    # plt.title('Correlator of Pi0 against M11 of quark flavour basis matrix')
    # plt.legend()
    # plt.savefig('Pi0M11_c.png')

    # plt.figure()
    # plt.errorbar(t, mean_M11_os[1:], yerr=err_M11_os[1:], label='M11 (os)')
    # plt.errorbar(t, mean_M22_os[1:], yerr=err_M22_os[1:], label='M22 (os)') 
    # plt.xlabel('t')
    # plt.ylabel('Correlators')
    # plt.title('Correlator of M11 and M22 of octet-singlet basis matrix')
    # plt.legend()
    # plt.savefig('M11M22os_c.png')

    # Comparing the different between the ll effective mass and the pion effective mass
    dmeff_samples = meff_ll_samples - meff_pion_samples
    mean_dmeff = np.nanmean(dmeff_samples, axis=0)
    err_dmeff = corr.bootstrap_std(dmeff_samples)

    # plt.figure()
    # plt.errorbar(t, mean_dmeff[1:-1], yerr=err_dmeff[1:-1])
    # plt.xlabel('t')
    # plt.ylabel('Difference')
    # plt.title('Difference between M11 meff and pion meff')
    # plt.savefig('Pi0M11_dMeff')

if __name__ == "__main__":
    main()