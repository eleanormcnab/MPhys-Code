
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from correlator_tools import CorrelatorTools
import warnings
warnings.filterwarnings("ignore")
plt.close('all')


def main():
    corr = CorrelatorTools(a = 0.1141, K = 1000, L = 16)

    # Define the list of flavours to run
    flavours_to_run = ['ll', 'sl', 'ss', 'ls']

    # Lists to store results for all flavours
    all_t = []
    all_db_mean = []
    all_db_err = []
    all_db_samples = []

    # Lists for split-source results
    all_db_split_mean = []
    all_db_split_err = []
    all_db_split_samples = []

    # For n = 2, 4, 8, 16, 32, 64 sources plots
    n_list = [2, 4, 8, 16, 32, 64]
    all_n_means = []
    all_n_errors = []

    for flavour in flavours_to_run:
        print(f"Flavour: {flavour}")

        # Empty lists for all configurations
        all_disco = []
        all_disco_imag = []

        all_disco_split = []
        all_disco_split_imag = []

        # Store correlators for different numbers of sources
        disco_by_n = [[] for n in n_list]

        # First and second loop flavours
        q1 = flavour[0]
        q2 = flavour[1]

        # Sources
        ihits = list(range(64))

        for conf in range(1500, 2661, 40):

            # Empty lists for each config, one loop per source
            loops_q1 = []
            loops_q2 = []

            for ihit in ihits:
                filename1 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc64/Z2Loop_' + q1 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
                t, loop1 = np.loadtxt(filename1, unpack=True, dtype=complex)
                loops_q1.append(loop1)

                filename2 = '/mnt/c/Git/MPhysCode/datafiles/h5datafile_disc64/Z2Loop_' + q2 + '_ihit' + str(ihit) + '.' + str(conf) + '.dat'
                t, loop2 = np.loadtxt(filename2, unpack=True, dtype=complex)
                loops_q2.append(loop2)

            loops_q1 = np.array(loops_q1)
            loops_q2 = np.array(loops_q2)

            # Calculate disconnected correlator for each number of sources
            for i, n in enumerate(n_list):
                if q1 == q2:
                    D = corr.disco_correlator(loops_q1[:n])
                else:
                    D = corr.disco_correlator(loops_q1[:n], loops_q2[:n])

                disco_by_n[i].append(D.real)

            # Original correlator using all 64 sources
            if q1 == q2:
                D = corr.disco_correlator(loops_q1)
            else:
                D = corr.disco_correlator(loops_q1, loops_q2)

            all_disco.append(D.real)
            all_disco_imag.append(D.imag)

            # Calculate split-source correlator
            if q1 == q2:
                D_split = corr.disco_correlator_split(loops_q1)
            else:
                D_split = corr.disco_correlator_split(loops_q1, loops_q2)

            all_disco_split.append(D_split.real)
            all_disco_split_imag.append(D_split.imag)

        # Fold both disconnected correlators
        folded_disco = corr.twoptf_c(all_disco)
        folded_disco_imag = corr.twoptf_c(all_disco_imag)

        folded_disco_split = corr.twoptf_c(all_disco_split)
        folded_disco_split_imag = corr.twoptf_c(all_disco_split_imag)

        # Original mean and error
        db_mean, db_samples = corr.bootstrap(folded_disco)
        db_err = corr.bootstrap_std(db_samples)

        # Split-source mean and error
        db_split_mean, db_split_samples = corr.bootstrap(folded_disco_split)
        db_split_err = corr.bootstrap_std(db_split_samples)

        # Checking imaginary parts
        im_mean, im_samples = corr.bootstrap(folded_disco_imag)
        im_err = corr.bootstrap_std(im_samples)

        im_split_mean, im_split_samples = corr.bootstrap(folded_disco_split_imag)
        im_split_err = corr.bootstrap_std(im_split_samples)

        print(f"  Original Im[D] mean: {np.mean(im_mean)}")
        print(f"  Split Im[D] mean: {np.mean(im_split_mean)}")

        # Append results to lists
        all_t.append(t)

        all_db_mean.append(db_mean[2:-1])
        all_db_err.append(db_err[2:-1])
        all_db_samples.append(db_samples)

        all_db_split_mean.append(db_split_mean[2:-1])
        all_db_split_err.append(db_split_err[2:-1])
        all_db_split_samples.append(db_split_samples)

        # Bootstrap correlators for different numbers of sources
        n_means = []
        n_errors = []

        for disco in disco_by_n:
            folded = corr.twoptf_c(disco)
            mean, samples = corr.bootstrap(folded)
            error = corr.bootstrap_std(samples)

            n_means.append(mean)
            n_errors.append(error)

        all_n_means.append(n_means)
        all_n_errors.append(n_errors)

        # Plot different numbers of sources
        plt.figure()

        for i, n in enumerate(n_list):
            plt.errorbar(t[2:-1], n_means[i][2:-1],
                         yerr=n_errors[i][2:-1],
                         label=str(n) + ' sources', fmt='o-')

        plt.xlabel('t')
        plt.ylabel('Disconnected correlator')
        plt.title('Disconnected correlator ' + flavour)
        plt.legend()
        plt.savefig('Disco_sources_' + flavour + '.png')
        plt.close()

        # Plot original and split-source comparison
        plt.figure()

        plt.errorbar(t[2:-1], db_mean[2:-1],
                     yerr=db_err[2:-1],
                     label='Original', fmt='o')

        plt.errorbar(t[2:-1], db_split_mean[2:-1],
                     yerr=db_split_err[2:-1],
                     label='Split sources')

        plt.xlabel('t')
        plt.ylabel('D(t)')
        plt.title('Disconnected correlator ' + flavour)
        plt.legend()
        plt.savefig('Disco_split_' + flavour + '.png')
        plt.close()

    return (all_t, all_db_mean, all_db_err, all_db_samples,
            all_db_split_mean, all_db_split_err, all_db_split_samples,
            all_n_means, all_n_errors)


if __name__ == "__main__":
    main()
