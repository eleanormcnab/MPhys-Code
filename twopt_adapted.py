import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit


hbarc = 197.3269804

def shifted_point_source(c, t_source):
    """Use PBC: C(t) = C(t + t_source)"""
    return np.roll(c, -t_source)


def source_average(all_c, t_sources):
    """Shift each two-point function to put its source at t = 0, then average over the sources"""
    shifted = [shifted_point_source(c, ts) for c, ts in zip(all_c, t_sources)]
    return np.mean(shifted, axis=0)

def z_files(c, L=16):
    """All "Z2" data files must be divided by the lattice volume"""
    return c / (L**3)

def bootstrap(configurations, K):
    """Bootstrap method for mean, variance, and sd for an observable over a list of configurations"""
    configurations = np.array(configurations)
    # N gluon configurations
    N = len(configurations)
    # Gauge average of each bootstrap sample
    bootstrap_averages = []
    # We have K sets of the data, from each subset within K- "k"- we pull a random number N times with replacement.
    for k in range(K):
        random_configurations = np.random.randint(0, N, N)
        # Gauge sample of the new bootstrap configurations
        average_k = np.mean(configurations[random_configurations], axis=0)
        bootstrap_averages.append(average_k)
    bootstrap_averages = np.array(bootstrap_averages)
    # O_tilde is the sample mean
    O_tilde = (1/K) * np.sum(bootstrap_averages, axis=0)
    return O_tilde, bootstrap_averages


def bootstrap_std(samples):
    """Bootstrap error on a quantity such as effective mass. The quantity is computed
    on each of the K subsets. We calculate the standard deviation across subsets."""
    samples = np.array(samples)
    K = len(samples)
    mean = np.sum(samples, axis=0) / K
    return np.sqrt(np.sum((samples - mean)**2, axis=0) / K)


def folded(c):
    "Fold two-point correlation function using periodic boundary conditions, C(t)=C(T-t)."
    T = len(c)
    cfolded = np.zeros(T)
    # PBC: t = 0 = t = T
    cfolded[0] = c[0]
    for t in range(1, T):
        next = (T - t)
        cfolded[t] = 0.5 * (c[t] + c[next])
    return cfolded


def twoptf_c(all_twopt):
    """List of folded C(t)s"""
    return [folded(c) for c in all_twopt]


def meff(c):
    """Effective mass from cosh(m) = (C(t-1) + C(t+1)) / (2 C(t))"""
    T = len(c)
    meff_c = np.zeros(T)
    # C(T) = C(0), so do not compute t = T - 1 as it only has one neighbour
    for t in range(1, T-1):
        previous = t - 1
        after = t + 1
        meff_c[t] = np.arccosh(0.5 * (c[previous] + c[after]) / c[t])
    return meff_c
    
    
def physical_mass(meff_values, t_min, t_max, a):
    """Average m_eff in MeV"""
    mass_sum = 0
    for t in range(t_min, t_max):
        mass_sum += meff_values[t]
    mean_mass = mass_sum / len(range(t_min, t_max))
    return mean_mass * hbarc / a


def main():
    # Lattice spacing in fm
    a = 0.12
    # Bootstrap subsets
    K = 1000
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

    for conf in range(1500, 2701, 40):
        # Empty list for this config
        c_this_config = []

        # Source times to get each file's data
        for ts in [0, 4, 8, 12, 16, 20, 24, 28]:
            filename = 'h5datafile_64/' + source + str(ts) + '_' + flavour + '.' + str(conf) + '.dat'
            t, twopt = np.loadtxt(filename, unpack=True)

            # Only the Z2 files are divided by the lattice volume
            if source == source2:
                twopt = z_files(twopt)
            
            c_this_config.append(twopt)

        # Once per config, after all the source times are loaded
        all_twopt.append(source_average(c_this_config, [0, 4, 8, 12, 16, 20, 24, 28]))
    folded_twopt = twoptf_c(all_twopt)
   

    # Mean and error of C(t)
    cb_mean, cb_samples = bootstrap(folded_twopt, K)

    # Effective mass of every bootstrap sample
    meff_samples = np.array([meff(c) for c in cb_samples])
    # Effective masses using both jackknife and bootstrap methods, and bootstrap errors on each m_eff
    meff_b = meff_samples.mean(axis=0)
    meff_b_err = bootstrap_std(meff_samples)

    
    # Plots of the average two-point function
    plt.figure()
    plt.semilogy(t, cb_mean)
    plt.xlabel('t')
    plt.ylabel('C_mean')
    plt.title('C_mean against time')
    plt.show()
    
    # Plot of the effective mass with bootstrap error bars
    # Start at t = 2 so the plot is symmetric
    plt.errorbar(t[2:-1], meff_b[2:-1], yerr=meff_b_err[2:-1])
    plt.xlabel('t')
    plt.ylabel('m_eff')
    plt.title("Bootstrap effective mass against time")
    plt.show()


    # Mass in MeV for with bootstrap error: WRONG RANGE WILL FIX
    mass_MeV_b = physical_mass(meff_b, 25, 71, a)
    mass_samples = np.array([physical_mass(m, 25, 71, a) for m in meff_samples])
    mass_MeV_b_err = bootstrap_std(mass_samples)
    
    # Print bootstrap method mass and error
    print(f"Mass from bootstrap method = {mass_MeV_b} +/- {mass_MeV_b_err} MeV")


if __name__ == "__main__":
    main()