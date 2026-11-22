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
    # Combining sample by sample then calculating error on summed terms

    # Elements M11, M22, M12, M21, of quark flavour basis matrix
    M11_qf_samples =  all_cb_samples[0] - 2.0 * all_db_samples[0]
    M22_qf_samples = all_cb_samples[2] - all_db_samples[2]
    M12_qf_samples =  -1* (np.sqrt(2)) * all_db_samples[1]
    M21_qf_samples = -1* (np.sqrt(2)) * all_db_samples[3]
        
    # Pion twopoint function in quark flavour basis
    pion_qf_samples = all_cb_samples[0]

    # Mean and bootstrap error
    mean_ll = np.mean(M11_qf_samples, axis=0)
    err_ll = corr.bootstrap_std(M11_qf_samples)
    mean_ss = np.mean(M22_qf_samples, axis=0)
    err_ss = corr.bootstrap_std(M22_qf_samples)
    mean_pion = np.mean(pion_qf_samples, axis=0)
    err_pion = corr.bootstrap_std(pion_qf_samples)
    mean_ls = np.mean(M12_qf_samples, axis=0)
    err_ls = corr.bootstrap_std(M12_qf_samples)
    mean_sl = np.mean(M21_qf_samples, axis=0)
    err_sl = corr.bootstrap_std(M21_qf_samples)

    #Combined effective masses, combined bootstrap errors
    meff_ll_samples = np.array([corr.num_meff(sample) for sample in M11_qf_samples])
    mean_meff_ll = np.nanmean(meff_ll_samples, axis=0)
    err_meff_ll = corr.bootstrap_std(meff_ll_samples)

    meff_ss_samples = np.array([corr.num_meff(sample) for sample in M22_qf_samples])
    mean_meff_ss = np.nanmean(meff_ss_samples, axis=0)
    err_meff_ss = corr.bootstrap_std(meff_ss_samples)

    meff_ls_samples = np.array([corr.num_meff(sample) for sample in M12_qf_samples])
    mean_meff_ls = np.nanmean(meff_ls_samples, axis=0)
    err_meff_ls = corr.bootstrap_std(meff_ls_samples)

    meff_sl_samples = np.array([corr.num_meff(sample) for sample in M21_qf_samples])
    mean_meff_sl = np.nanmean(meff_sl_samples, axis=0)
    err_meff_sl = corr.bootstrap_std(meff_sl_samples)

    meff_pion_samples = np.array([corr.num_meff(sample) for sample in pion_qf_samples])
    mean_meff_pion = np.nanmean(meff_pion_samples, axis=0)
    err_meff_pion = corr.bootstrap_std(meff_pion_samples)

    # Elements M11, M22, M21, M12, of octet-singlet basis matrix
    M21_os_samples = 1/(3 *np.sqrt(2)) * (4.0 * all_db_samples[0] - 2.0 * all_cb_samples[0] - 4.0 * all_db_samples[3] + 2.0 * all_db_samples[1] - 2.0 * all_db_samples[2] + 2.0 * all_cb_samples[2])
    mean_M21_os = np.mean(M21_os_samples, axis=0)
    err_M21_os = corr.bootstrap_std(M21_os_samples)
    plateau_M21_os = np.mean(mean_M21_os[12:20])

    M12_os_samples = 1/(3 *np.sqrt(2)) * (4.0 * all_db_samples[0] - 2.0 * all_cb_samples[0] + 2.0 * all_db_samples[3] - 4.0 * all_db_samples[1] - 2.0 * all_db_samples[2] + 2.0 * all_cb_samples[2])
    mean_M12_os = np.mean(M12_os_samples, axis=0)
    err_M12_os = corr.bootstrap_std(M12_os_samples)
    plateau_M12_os = np.mean(mean_M12_os[12:20])
 
    M11_os_samples = 1/6 * (4.0 * all_db_samples[0] - 2.0 * all_cb_samples[0] - 4.0 * all_db_samples[3] - 4.0 * all_db_samples[1] + 4.0 * all_db_samples[2] - 4.0 * all_cb_samples[2])
    M22_os_samples = 1/3 * (4.0 * all_db_samples[0] - 2.0 * all_cb_samples[0] + 2.0 * all_db_samples[3] + 2.0 * all_db_samples[1] + all_db_samples[2] - all_cb_samples[2])
    # Combined effective masses, combined bootstrap errors
    meff_M21_os_samples = np.array([corr.num_meff(sample) for sample in M21_os_samples])
    mean_meff_M21_os = np.nanmean(meff_M21_os_samples, axis=0)
    err_meff_M21_os = corr.bootstrap_std(meff_M21_os_samples)
    meff_M12_os_samples = np.array([corr.num_meff(sample) for sample in M12_os_samples])
    mean_meff_M12_os = np.nanmean(meff_M12_os_samples, axis=0)
    err_meff_M12_os = corr.bootstrap_std(meff_M12_os_samples)

    
    # Plots of effective masses
    plt.figure()
    plt.errorbar(t, mean_meff_ll[1:-1], yerr=err_meff_ll[1:-1], label='M11 (ll)')
    plt.errorbar(t, mean_meff_pion[1:-1], yerr=err_meff_pion[1:-1], label='Pion')
    plt.xlabel('t')
    plt.ylabel('Effective Mass of Pi0 against M11 of quark flavour basis matrix')
    plt.title('Effective Masses')
    plt.legend()
    plt.savefig('Pi0M11_meff.png')

    plt.figure()
    plt.errorbar(t, mean_meff_M21_os[1:-1], yerr=err_meff_M21_os[1:-1], label='M11')
    plt.errorbar(t, mean_meff_M12_os[1:-1], yerr=err_meff_M12_os[1:-1], label='m22')
    plt.xlabel('t')
    plt.ylabel('Effective Mass of M11 and M22 of octet-singlet basis matrix')
    plt.title('Effective Masses')
    plt.legend()
    plt.savefig('M11M22os_meff.png')

    plt.figure()
    plt.errorbar(t, mean_meff_ls[1:-1], yerr=err_meff_ls[1:-1], label='M12 (ls)')
    plt.errorbar(t, mean_meff_sl[1:-1], yerr=err_meff_sl[1:-1], label='M21 (sl)')
    plt.xlabel('t')
    plt.ylabel('Effective Mass of M12 and M21 of quark flavour basis matrix')
    plt.title('Effective Masses')
    plt.legend()
    plt.savefig('M12M21_qf_meff.png')

    
   # Plotting twopoint functions-I took a splice of C(t) results hence the full range
    plt.figure()
    plt.errorbar(np.arange(1, 32), mean_ll[1:], yerr=err_ll[1:], label='M11 (ll)')
    plt.errorbar(np.arange(1, 32), mean_pion[1:], yerr=err_pion[1:], label='Pion')
    plt.xlabel('t')
    plt.ylabel('Correlators')
    plt.title('Correlator of Pi0 against M11 of quark flavour basis matrix')
    plt.legend()
    plt.savefig('Pi0M11_c.png')

    plt.figure()
    plt.errorbar(np.arange(1, 32), mean_M21_os[1:], yerr=err_M21_os[1:], label='M11 (os)')
    plt.errorbar(np.arange(1, 32), mean_M12_os[1:], yerr=err_M12_os[1:], label='M22 (os)') 
    plt.xlabel('t')
    plt.ylabel('Correlators')
    plt.title('Correlator of M11 and M22 of octet-singlet basis matrix')
    plt.legend()
    plt.savefig('M11M22os_c.png')

    # Comparing the different between the ll effective mass and the pion effective mass
    dmeff_samples = meff_ll_samples - meff_pion_samples
    mean_dmeff = np.nanmean(dmeff_samples, axis=0)
    err_dmeff = corr.bootstrap_std(dmeff_samples)

    plt.figure()
    plt.errorbar(t, mean_dmeff[1:-1], yerr=err_dmeff[1:-1])
    plt.xlabel('t')
    plt.ylabel('Difference')
    plt.title('Difference between M11 meff and pion meff')
    plt.savefig('Pi0M11_dMeff')

    # Plateau meff values of M21 and M12 matrix elements
    # [:, 12:20] selects all samples (:) and timesteps 12-20, axis=1 averages over time for each sample
    plateau_M21_samples = np.nanmean(meff_M21_os_samples[:, 12:20], axis=1)
    plateau_meff_M21_os = np.nanmean(plateau_M21_samples)
    plateau_meff_M21_os_err = corr.bootstrap_std(plateau_M21_samples)

    plateau_M12_samples = np.nanmean(meff_M12_os_samples[:, 12:20], axis=1)
    plateau_meff_M12_os = np.nanmean(plateau_M12_samples)
    plateau_meff_M12_os_err = corr.bootstrap_std(plateau_M12_samples)
        
    print(f'M21 matrix element: {plateau_meff_M21_os} +/- {plateau_meff_M21_os_err}')
    print(f'M12 matrix element: {plateau_meff_M12_os} +/- {plateau_meff_M12_os_err}')
  
if __name__ == "__main__":
    main()
