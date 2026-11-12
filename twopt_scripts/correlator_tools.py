import numpy as np
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
plt.close('all')

class CorrelatorTools:
    hbarc = 197.3269804

    def __init__(self, a, K, L):
        """Defining lattice spacing, length, and number of bootstrap resamples"""
        self.a = a
        self.K = K
        self.L = L
    def shifted_point_source(self, c, t_source):
        """Use PBC: C(t) = C(t + t_source)"""
        return np.roll(c, -t_source)

    
    def source_average(self, all_c, t_sources):
        """After shifting each two-point function to put its source at t = 0, average over the sources"""
        # Averaging helps deal with NaNs: noisy values make C(t) sometimes less than 1, below which arccosh in undefined.
        shifted = [self.shifted_point_source(c, ts) for c, ts in zip(all_c, t_sources)]
        return np.mean(shifted, axis=0)

        
    def z_files(self, c, L=16):
        """All "Z2" data files must be divided by the lattice volume"""
        return c / (self.L**3)

    def bootstrap(self, configurations):
        """Bootstrap method for mean, variance, and sd for an observable over a list of configurations"""
        configurations = np.array(configurations)
        # N gluon configurations
        N = len(configurations)
        # Gauge average of each bootstrap sample
        bootstrap_averages = []
        # We have K sets of the data, from each subset within K- "k"- we pull a random number N times with replacement.
        for k in range(self.K):
            random_configurations = np.random.randint(0, N, N)
            # Gauge sample of the new bootstrap configurations
            average_k = np.mean(configurations[random_configurations], axis=0)
            bootstrap_averages.append(average_k)
        bootstrap_averages = np.array(bootstrap_averages)
        # O_tilde is the sample mean
        O_tilde = (1/self.K) * np.sum(bootstrap_averages, axis=0)
        return O_tilde, bootstrap_averages


    def bootstrap_std(self, samples):
        """Bootstrap error on a quantity such as effective mass. The quantity is computed
        on each of the K subsets. We calculate the standard deviation across subsets."""
        samples = np.array(samples)
        n = len(samples)
        mean = np.sum(samples, axis=0) / n
        return np.sqrt(np.sum((samples - mean)**2, axis=0) / n)


    def folded(self, c):
        "Fold two-point correlation function using periodic boundary conditions, C(t)=C(T-t)."
        T = len(c)
        cfolded = np.zeros(T)
        # PBC: t = 0 = t = T
        cfolded[0] = c[0]
        for t in range(1, T):
            next = (T - t)
            cfolded[t] = 0.5 * (c[t] + c[next])
        return cfolded


    def twoptf_c(self, all_twopt):
        """List of folded C(t)s"""
        return [self.folded(c) for c in all_twopt]


    def meff(self, c):
        """Effective mass from cosh(m) = (C(t-1) + C(t+1)) / (2 C(t))"""
        T = len(c)
        meff_c = np.zeros(T)
        # C(T) = C(0), so do not compute t = T - 1 as it only has one neighbour
        for t in range(1, T-1):
            previous = t - 1
            after = t + 1
            meff_c[t] = np.arccosh(0.5 * (c[previous] + c[after]) / c[t])
        return meff_c
        
        
    def physical_mass(self, meff_values, t_min, t_max):
        """Average m_eff in MeV"""
        mass_sum = 0
        for t in range(t_min, t_max):
            mass_sum += meff_values[t]
        mean_mass = mass_sum / len(range(t_min, t_max))
        return mean_mass * CorrelatorTools.hbarc / self.a

    def disco_correlator(self, loops, loops2=None):
        """Calculating correlators for disconnected loops.
        loops2=None gives the same-flavour correlator (ll or ss);
        passing a second array gives the mixed one."""
        # loops[i, x0] = L_i(x0), noise source summed over spatial position
        # N = number of sources, T = number of time slices
        N, T = loops.shape
        # Same flavour unless a second array is given
        if loops2 is None:
            loops2 = loops
        # D = disconnected correlator
        D = np.zeros(T)
        # t = x0 - y0
        for t in range(T):
            pairs = []
            for x0 in range(T):
                y0 = (x0 - t)
                sum_i = loops[:, x0].sum()
                sum_j = loops2[:, y0].sum()
                all_pairs = sum_i * sum_j

                # Autocorrelators should be subtracted
                same_source = (loops[:, x0] * loops2[:, y0]).sum()

                pairs.append((all_pairs - same_source) / (N * (N - 1)))
            # Average over x0, then divide by L^3
            D[t] = np.mean(pairs) / self.L**3
        return D
