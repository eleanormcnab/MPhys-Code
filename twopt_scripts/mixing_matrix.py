import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from correlator_tools import CorrelatorTools
import disco_twopt as disconnected
import twopt_adapted as connected
plt.close('all')

# TO FIX: Bootstrap sampling combination
def main():
    corr = CorrelatorTools(a = 0.1141, K = 1000, L = 16)

    # Connected and disconnected results
    all_t_d, all_db_mean, all_db_err, all_db_samples = disconnected.main()
    all_t, all_cb_mean, all_cb_err, all_masses_n, all_masses_err_n, all_masses, all_masses_err, all_cb_samples = connected.main()

    t = all_t[0]

    # Elements M11 and M22 of quark flavour basis matrix
    M11_qf = all_cb_samples[0] - 2.0 * all_db_samples[0]
    M22_qf = all_cb_samples[2] - 2.0 * all_db_samples[2]

    # Mean and bootstrap error
    mean_ll = np.mean(M11_qf, axis=0)
    err_ll = corr.bootstrap_std(M11_qf)
    mean_ss = np.mean(M22_qf, axis=0)
    err_ss = corr.bootstrap_std(M22_qf)

    plt.figure()
    plt.errorbar(t, mean_ll, yerr=err_ll)
    plt.xlabel('t')
    plt.ylabel('C - 2D')
    plt.title('Combined correlator ll against time')
    plt.show()

    plt.figure()
    plt.errorbar(t, mean_ss, yerr=err_ss)
    plt.xlabel('t')
    plt.ylabel('C - 2D')
    plt.title('Combined correlator ss against time')
    plt.show()

    # Arrays start at t = 2, so t = 12 to 20 is indices 10 to 18
    print(f"Plateau (ll) = {np.mean(mean_ll[10:19])}")
    print(f"Plateau (ss) = {np.mean(mean_ss[10:19])}")

if __name__ == "__main__":
    main()