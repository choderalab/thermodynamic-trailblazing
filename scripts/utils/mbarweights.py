#!/usr/env/python

import logging
import os

import numpy as np
from scipy.special import logsumexp
from yank.analyze import YankReplicaExchangeAnalyzer


logger = logging.getLogger(__name__)


# ============================================================
# ANALYZER IMPLEMENTING NEW DECORRELATION METHOD
# ============================================================

class DoublySelfConsistentAnalyzer(YankReplicaExchangeAnalyzer):

    def _get_equilibration_data(self, replica_energies=None, neighborhoods=None, replica_state_indices=None,
                                equil_max_n_iterations=10):
        """Implement the doubly self-consistent automatic detection of equilibration and statistical inefficiency.

        Parameters
        ----------
        equil_max_n_iterations : int, optional, default=10
            After ``equil_max_n_iterations``, if the equilibration detection
            hasn't converged, an error is raised.

        """
        # TODO: This doesn't work with restraint reweighting.
        import pymbar

        # Check if we need to read the data.
        if replica_energies is None or replica_state_indices is None:
            # Case where no input is provided
            replica_energies, _, neighborhoods, replica_state_indices = self._read_energies(
                truncate_max_n_iterations=True)

        # replica_energies[r][l][n] is the reduced potential sampled by REPLICA r
        # and evaluated at state l at iteration n. We need to deconvolute to kln
        # format (i.e. where the first index is state k rather than replica r).
        n_replicas, n_states, n_iterations = replica_energies.shape
        mbar_energies = np.empty(shape=(n_states, n_states, n_iterations), dtype=np.float64)
        for iteration in range(n_iterations):
            state_indices = replica_state_indices[:, iteration]
            mbar_energies[state_indices,:,iteration] = replica_energies[:,:,iteration]

        # Start iterative algorithm.
        for iteration in range(equil_max_n_iterations):
            logger.debug(f'Equilibration detection iteration {iteration}')

            N_k = np.array([mbar_energies.shape[-1] for _ in range(mbar_energies.shape[0])])
            logger.debug(f'    Initializing MBAR')
            mbar = pymbar.MBAR(mbar_energies, N_k)
            # We don't compute the covariance to make it faster.
            logger.debug(f'    Computing MBAR free energy')
            Deltaf_ij, _, _ = mbar.getFreeEnergyDifferences(compute_uncertainty=False, return_theta=False)

            # Print the difference in free energy.
            Deltaf = Deltaf_ij[0]
            logger.debug(f'    Deltaf (before discarding) = {Deltaf[-1]}')

            # Compute the equilibration time and statistical
            # inefficiency of the bound and decoupled states.
            logger.debug(f'    Computing MBAR weights')
            mbar_log_weight_trajectories = compute_hrex_log_weight_trajectory(
                mbar_energies, Deltaf, computed_states=[0, n_states-1], verbose=False)

            # Perform automatic equilibration detection on the MBAR weight time series.
            # The normalized correlation function is invariant under scaling factors.
            mbar_weight_trajectories = compute_scaled_mbar_weight_trajectories(mbar_log_weight_trajectories)
            equilibration_data = np.empty(shape=(len(mbar_weight_trajectories), 3))
            logger.debug(f'    Running detectEquilibration')
            for trajectory_idx, mbar_weight_trajectory in enumerate(mbar_weight_trajectories):
                equilibration_data[trajectory_idx] = pymbar.timeseries.detectEquilibration(mbar_weight_trajectory, fast=False, nskip=1)

            for state_idx in range(len(equilibration_data)):
                t, g_t, N_eff_max = equilibration_data[state_idx]
                logger.debug(f'    state {state_idx}: n_equil_iterations={t}, statistical_ineff={g_t}, n_effective_iter={N_eff_max}')

            # Subsample mbar_energies.
            t_max = max(equilibration_data[:,0])
            g_t_max = max(equilibration_data[:,1])
            uncorrelated_indices = []
            n = 0
            while int(round(n * g_t_max)) < mbar_energies.shape[-1] - t_max:
                t = int(round(n * g_t_max))
                # ensure we don't sample the same point twice
                if (n == 0) or (t != uncorrelated_indices[n - 1]):
                    uncorrelated_indices.append(t)
                n += 1
            uncorrelated_indices = np.array(uncorrelated_indices, dtype=np.int) + int(t_max)
            n_effective_max = len(uncorrelated_indices)
            logger.debug(f'    discarding a total of {mbar_energies.shape[-1] - n_effective_max} iterations '
                         f'({n_effective_max} iterations remaining)')
            mbar_energies = mbar_energies[:,:,uncorrelated_indices]

            # Check if the algorithm converged.
            if t_max == 0 and g_t_max < 2:
                self._equilibration_data = tuple([t_max, g_t, n_effective_max])
                logger.debug('Doubly self-consistent equilibration data:')
                logger.debug('    number of iterations discarded to equilibration : {}'.format(t_max))
                logger.debug('    statistical inefficiency of production region   : {}'.format(g_t))
                logger.debug('    effective number of uncorrelated samples        : {}'.format(n_effective_max))

                return t_max, g_t, n_effective_max

        # If we get to this point, we've run through the
        # maximum number of iterations without converging.
        raise RuntimeError(f'Ran {equil_max_n_iterations} without converging doubly self-consistent analysis.')


# =============================================================================
# CORRELATION TIMES
# =============================================================================

def compute_sample_mbar_log_weight(sample_states_energies, Deltaf, computed_states=None):
    """Compute the MBAR log weight of a single HREX replica sample at the computed states.

    The weight for state i and sample x is given by

                      e^(-u_i(x))
    w_i(x) = -----------------------------
              sum_j( e^(Df_j - u_j(x)) / K )

    where K is the total number of states, Df_j and u_j are the free energy
    difference and reduced potential of the sample for state j.

    Parameters
    ----------
    sample_states_energies : numpy.ndarray
        sample_states_energies[l] is the reduced potential (in kT) of
        the sample evaluated at state l.
    Deltaf : numpy.ndarray
        Deltaf[k] is the MBAR unitless free energy difference between
        state k and an arbitrary reference.
    computed_states : Iterable[int]
        If given, only the weights of the states indexed by the elements
        of computed_states are computed. Otherwise, the weights for all
        states are given.

    Returns
    -------
    states_mbar_log_weights : numpy.ndarray
        states_mbar_log_weights[i] is log(w_j(x)) with j = computed_states[i]
        if computed_states is given or j = i otherwise.

    """
    n_states = sample_states_energies.shape[0]
    if computed_states is None:
        computed_states = list(range(n_states))
    states_mbar_log_weights = np.empty(shape=len(computed_states))

    for i, k in enumerate(computed_states):
        states_mbar_log_weights[i] = - logsumexp(Deltaf + sample_states_energies[k] - sample_states_energies) + np.log(n_states)
    return states_mbar_log_weights


def compute_hrex_mbar_log_weight(iteration_mbar_energies, Deltaf, computed_states=None):
    """Compute the MBAR log weight of a single HREX iteration at the computed states.

    The weight for the iteration for state i is given by

    sum_k( w_i(x_k) )

    where the sum is taken over all the samples collected from the parallel replicas.

    Parameters
    ----------
    iteration_mbar_energies : numpy.ndarray
        iteration_mbar_energies[k][l] is the reduced potential (in kT) sampled
        from state k and evaluated at state l.
    Deltaf : numpy.ndarray
        Deltaf[k] is the MBAR unitless free energy difference between
        state k and an arbitrary reference.
    computed_states : Iterable[int]
        If given, only the weights of the states indexed by the elements
        of computed_states are computed. Otherwise, the weights for all
        states are given.

    Returns
    -------
    iteration_mbar_log_weights : numpy.ndarray
        iteration_mbar_log_weights[i] is log(sum_k( w_j(x_k) )) with
        j = computed_states[i] if computed_states is given or j = i otherwise.

    See Also
    --------
    compute_sample_mbar_log_weight

    """
    n_states = iteration_mbar_energies.shape[0]
    if computed_states is None:
        computed_states = list(range(n_states))
    n_computed_states = len(computed_states)
    samples_mbar_log_weights = np.zeros(shape=(n_states, n_computed_states))

    for sampled_state_idx, sample_states_energies in enumerate(iteration_mbar_energies):
        samples_mbar_log_weights[sampled_state_idx] = compute_sample_mbar_log_weight(
            sample_states_energies, Deltaf, computed_states)

    # Sum all sample weights for each state.
    iteration_mbar_log_weights = np.empty(shape=n_computed_states)
    for k in range(n_computed_states):
        iteration_mbar_log_weights[k] = logsumexp(samples_mbar_log_weights[:,k]) - np.log(n_states)
    return iteration_mbar_log_weights


def compute_hrex_log_weight_trajectory(mbar_energies, Deltaf, computed_states=None, verbose=False):
    """Compute the trajectory of MBAR log weights generated by the HREX calculation.

    Parameters
    ----------
    mbar_energies : np.ndarray
        mbar_energies[k][l][n] is the reduced potential sampled at state k and
        evaluated at state l at iteration n.
    Deltaf : np.ndarray
        Deltaf[i] is the free energy difference between state i and j computed
        as f_i - f_0.
    computed_states : Iterable[int]
        If given, only the weights of the states indexed by the elements
        of computed_states are computed. Otherwise, the weights for all
        states are given.

    Returns
    -------
    hrex_log_weight_trajectories : np.ndarray
        hrex_log_weight_trajectories[k][n] is the MBAR log weight generated by the
        HREX simulation at iteration n for state j = computed_states[k], if
        computed_states is given, or j = k otherwise.

    See Also
    --------
    compute_sample_mbar_log_weight
    compute_hrex_mbar_log_weight

    """
    n_states, _, n_iterations = mbar_energies.shape
    if computed_states is None:
        computed_states = list(range(n_states))
    n_computed_states = len(computed_states)
    hrex_mbar_log_weight_trajectories = np.empty(shape=(n_computed_states, n_iterations))
    for i in range(n_iterations):
        if verbose:
            print(f'\rIteration {i}/{n_iterations}', end='')
        hrex_mbar_log_weight_trajectories[:,i] = compute_hrex_mbar_log_weight(
            mbar_energies[:,:,i], Deltaf, computed_states)
    if verbose:
        print()
    return hrex_mbar_log_weight_trajectories


def compute_scaled_mbar_weight_trajectories(mbar_log_weight_trajectories):
    """Computed a scaled version of the weights from the log weights.

    The scaling is applied so that the most relevant weights will not
    overflow. Properties like the normalized correlation function are
    preserved under scaling.

    Parameters
    ----------
    mbar_log_weight_trajectories : np.ndarray
        mbar_log_weight_trajectories[k][n] is the MBAR log weight generated by the
        HREX simulation at iteration n for state k.

    Returns
    -------
    mbar_weight_trajectories : np.ndarray
        mbar_log_weight_trajectories[k][n] is the MBAR weight generated by the
        HREX simulation at iteration n for state k up to an arbitrary scaling
        factor.

    """
    mbar_weight_trajectories = np.empty(shape=mbar_log_weight_trajectories.shape)
    for i, mbar_log_weight_trajectory in enumerate(mbar_log_weight_trajectories):
        mbar_weight_trajectories[i] = np.exp(mbar_log_weight_trajectory - np.max(mbar_log_weight_trajectory))
    return mbar_weight_trajectories


def plot_weight_trajectory(mbar_energies):
    from pymbar import timeseries

    file_path = os.path.join('..','YankAnalysis', 'SamplingEnergies', 'CB8-G3-0-complex-all.npz')
    mbar_energies = np.load(file_path)['arr_0']

    # Load the maximum likelihood estimators of the free energies.
    file_path = os.path.join('..','YankAnalysis', 'SamplingEnergies', 'Deltaf.npy')
    # Remove the states used to reweight the restraint and the isotropic dispersion interactions.
    Deltaf = np.load(file_path)[0][2:-2]


    n_iterations = mbar_energies.shape[-1]
    computed_states = [0, len(Deltaf)-1]
    if computed_states[-1] != len(Deltaf)-1:
        computed_states.append(len(Deltaf)-1)
    mbar_energies = mbar_energies[:,:,:n_iterations]

    hrex_mbar_log_weight_trajectories = compute_hrex_log_weight_trajectory(
        mbar_energies, Deltaf, computed_states=computed_states)
    # np.save('temp.npy', hrex_mbar_log_weight_trajectories)
    # hrex_mbar_log_weight_trajectories = np.load('temp.npy')

    # The normalized autocorrelation function is insensitive to shifts in free
    # energy so we can make sure that the largest weights do not overflow.
    hrex_mbar_weight_trajectories = np.empty(shape=hrex_mbar_log_weight_trajectories.shape)
    for i, hrex_mbar_log_weight_trajectory in enumerate(hrex_mbar_log_weight_trajectories):
        hrex_mbar_weight_trajectories[i] = np.exp(hrex_mbar_log_weight_trajectories[i] - np.max(hrex_mbar_log_weight_trajectories[i]))

    fig, ax = plt.subplots(figsize=(9.8, 7))
    for i, state in enumerate(computed_states):
        ax.plot(hrex_mbar_weight_trajectories[i], label=str(state))

    # Compute statistical inefficiency of each weight trajectory.
    for i, hrex_mbar_weight_trajectory in enumerate(hrex_mbar_weight_trajectories):
        # We use multiple so that we can get the correlation function as well.
        g_t, C_t = timeseries.statisticalInefficiencyMultiple([hrex_mbar_weight_trajectory], fast=False,
                                                              return_correlation_function=True)
        print(i, g_t, g_t / n_iterations * 40000)
        t, g_t, Neff_max = timeseries.detectEquilibration(hrex_mbar_weight_trajectory, fast=False, nskip=1)
        print(i, t, g_t, Neff_max)
        print()

    # Check that the MBAR free energy and the logsumexp of the weights agree with each other.
    print(logsumexp(hrex_mbar_log_weight_trajectories[0]) - logsumexp(hrex_mbar_log_weight_trajectories[1]))
    print(Deltaf[-1] - Deltaf[0])

    # TODO Try to predict uncertainty with the given correlation time and error propagation
    # TODO: Try to predict uncertainty with the blocking method and propagation.

    # ax.set_xlim((-1, 68))
    ax.set_ylabel('HREX MBAR log weight')
    ax.set_xlabel('iteration')
    ax.legend()
    plt.tight_layout()
    plt.savefig('hrex_mbar_log_weights.pdf')
