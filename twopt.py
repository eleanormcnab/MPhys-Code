import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit


hbarc = 197.3269804


def jacknife(configurations):
    """Jackknife method for mean, variance, and sd for an observable over a list of configurations"""
    # N gluon configurations
    N = len(configurations)
    jacknife_subsets = []
    jacknife_variance = 0
    # One subset per configuration
    for n in range(N):
        # The nth subset has the nth configuration removed
        subset = configurations[:n] + configurations[n+1:]
        # Gauge average of each observable
        jacknife_subsets.append(np.mean((subset), axis = 0))  
    jacknife_subsets = np.array(jacknife_subsets)
    # O_hat for variance formula
    jacknife_mean = sum(jacknife_subsets)/(N)
    # Variance and SD formulae. x is a jackknife sample.
    for x in jacknife_subsets:
        jacknife_variance += (N - 1)/N *((x- jacknife_mean)**2)
    jacknife_sd = np.sqrt(jacknife_variance)
    return jacknife_mean, jacknife_variance, jacknife_sd


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
    # Variance and sd formulae
    bootstrap_variance = (1/K) * np.sum((bootstrap_averages - O_tilde)**2, axis=0)
    bootstrap_sd = np.sqrt(bootstrap_variance)
    return O_tilde, bootstrap_variance, bootstrap_sd, bootstrap_averages


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


# def num_meff(c):
#     """Effective mass from numerically solving C(t)/C(t+1)"""
#     T = len(c)
#     num_meff_c = np.zeros(T)
#     for t in range(1, T-1):
#         def numerical_function(meff):
#             return np.cosh(meff*(t - T/2)) / np.cosh(meff*(t + 1 - T/2)) - c[t]/c[t+1]
        
#         # Use fsolve to solve for meff
#         num_meff_c[t] = fsolve(numerical_function, 0.1)[0]
        
#     return num_meff_c
        
    
def physical_mass(meff_values, t_min, t_max, a):
    """Average m_eff in MeV"""
    mass_sum = 0
    for t in range(t_min, t_max):
        mass_sum += meff_values[t]
    mean_mass = mass_sum / len(range(t_min, t_max))
    return mean_mass * hbarc / a

def meff_constant(meff_values, t):
    """Find the effective mass by fitting a constant to the effective mass in the plateau region"""
    def f(t, m):
        return m
    optimised = curve_fit(f, t, meff_values, p0=0.2)
    # curve_fit returns an array, extract the effective mass entry
    return optimised[0][0]

def folded_leptonic(c):
    "For the leptonic two-point function, fold using periodic boundary conditions, C(t)=-C(T-t). Backward propagating direction has opposite sign to other C(t)."
    T = len(c)
    cfolded = np.zeros(T)
    # PBC: t = 0 = t = T
    cfolded[0] = c[0]
    for t in range(1, T):
        next = (T - t)
        cfolded[t] = 0.5 * (c[t] - c[next])
    return cfolded

def fpi(cA, c, m):
    """Decay constant of pion"""
    T = len(c)
    t = np.arange(T)
    plus = np.exp(-m * t) + np.exp(-m * (T - t))
    minus = np.exp(-m * t) - np.exp(-m * (T - t))
    # When the minus = 0 we have a divide by zero error. We ignore the warning and use nanmean later to ignore NaN.
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        result = np.sqrt((2 / m) * cA**2 / c * plus / minus**2)
    return result

def fpi_physical(fpi_values, t_min, t_max, a):
    """Average decay constant in MeV, ignoring undefined values"""
    mean_fpi = np.nanmean(fpi_values[t_min:t_max])
    return mean_fpi * hbarc / a

def main():
    # Number of configurations
    num_conf = 100
    # Lattice spacing in fm
    a = 0.12
    # Bootstrap subsets
    K = 1000
    # Empty lists for all configurations
    all_twopt = []
    all_twopt_leptonic = []

    # Loop over 100 files once per config
    for conf in range(num_conf):
        filename='data/pion_g5-g5.%s.dat' %(conf)
        t, twopt = np.loadtxt(filename, unpack=True)
        all_twopt.append(twopt)

    folded_twopt = twoptf_c(all_twopt)

    # Pion decay constant data extraction, looping over 100 files once per config
    for conf in range(num_conf):
        filename='data/pion_g5-g0g5.%s.dat' %(conf)
        t, twopt_leptonic = np.loadtxt(filename, unpack=True)
        all_twopt_leptonic.append(twopt_leptonic)

    folded_twopt_leptonic = [folded_leptonic(c) for c in all_twopt_leptonic]

    # Mean and error of C(t)
    c_mean, c_variance, c_sd = jacknife(folded_twopt)
    cb_mean, cb_variance, cb_sd, cb_samples = bootstrap(folded_twopt, K)

    # Effective mass of every bootstrap sample
    meff_samples = np.array([meff(c) for c in cb_samples])
    # Effective masses using both jackknife and bootstrap methods, and bootstrap errors on each m_eff
    meff_b = meff_samples.mean(axis=0)
    meff_b_err = bootstrap_std(meff_samples)
    meff_c = meff(c_mean)
    
    
    # Plots of the average two-point functions
    plt.figure()
    plt.semilogy(t, c_mean, label = "Jackknife")
    plt.semilogy(t, cb_mean, linestyle = ':', label = "Bootstrap")
    #plt.errorbar(t, c_mean, yerr = c_sd)
    plt.xlabel('t')
    plt.ylabel('C_mean')
    plt.title('C_mean against time')
    plt.legend(["Jackknife", "Bootstrap"])
    plt.show()

    # Plots of relative error on two-point functions
    plt.figure()
    plt.semilogy(t, c_sd/c_mean, label = "Jackknife")
    plt.semilogy(t, cb_sd/cb_mean, linestyle = ':', label = "Bootstrap")
    plt.xlabel('t')
    plt.ylabel('C_sd/C_mean')
    plt.title('Relative error on C(t) against time')
    plt.legend(["Jackknife", "Bootstrap"])
    plt.show()
    
    # Plots of the effective masses. Separate plots to see bootstrap error bars.
    plt.figure()
    # Start at t = 2 so the plot is symmetric
    plt.plot(t[2:-1], meff_c[2:-1], label = "Jackknife")
    plt.xlabel('t')
    plt.ylabel('m_eff')
    plt.title("Jackknife effective mass against time")
    plt.show()

    plt.figure()
    # Start at t = 2 so the plot is symmetric
    plt.errorbar(t[2:-1], meff_b[2:-1], yerr=meff_b_err[2:-1], label="Bootstrap")
    plt.xlabel('t')
    plt.ylabel('m_eff')
    plt.title("Bootstrap effective mass against time")
    plt.show()

    
    # Mass in MeV for Jackknife method
    mass_MeV = physical_mass(meff_c, 25, 71, a)

    # Mass in MeV for Bootstrap method
    mass_MeV_b = physical_mass(meff_b, 25, 71, a)
    mass_samples = np.array([physical_mass(m, 25, 71, a) for m in meff_samples])
    mass_MeV_b_err = bootstrap_std(mass_samples)
    
    #Print mass results, and error on bootstrap method mass
    print(f"Mass from jackknife method = {mass_MeV} MeV")
    print(f"Mass from bootstrap method = {mass_MeV_b} +/- {mass_MeV_b_err} MeV")

    # Second method: Find effective mass by fitting a constant to the effective mass in the plateau region
    m = meff_constant(meff_b[25:71], t[25:71])
    # Find error using bootstrap method
    fit_error = bootstrap_std([meff_constant(m_s[25:71], t[25:71]) for m_s in meff_samples]) * hbarc / a
    print(f"Constant fit mass = {m*hbarc/a} +/- {fit_error} MeV")

    
    # For decay constant
    # We have to pair C and C_A for each configuration for bootstrap sampling.
    pairs =[]
    for n in range(num_conf):
        pairs.append([folded_twopt[n], folded_twopt_leptonic[n]])

    f_mean, f_variance, f_sd, f_samples = bootstrap(pairs, K)
    mass = np.mean(meff_b[25:71])
    # From pairing, index 0 corresponds to C, index 1 to C_A
    fpi_values = fpi(f_mean[1], f_mean[0], mass)
    fpi_b = fpi_physical(fpi_values, 25, 71, a)

    # f_pi for each subset, calculated for bootstrap error 
    fpi_samples = []
    for k in range(K):
        # C and C_A of bootstrap k
        cP_k = f_samples[k][0]
        cA_k = f_samples[k][1]
        # f_pi at every time step for this subset
        fpi_k = fpi(cA_k, cP_k, mass)
        fpi_samples.append(fpi_physical(fpi_k, 25, 71, a))
    fpi_b_err = bootstrap_std(fpi_samples)

    print(f"Decay constant = {fpi_b} +/- {fpi_b_err}")



if __name__ == "__main__":
    main()