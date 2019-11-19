#!/usr/env/python


import numpy as np
import scipy.integrate


# =============================================================================
# UTILITIES
# =============================================================================

def read_states_energies(nc_file_path):
    """Return the MBAR energy matrix in kln form.

    Parameters
    ----------
    nc_file_path : str
        The path to the NetCDF file containing the energies.

    Returns
    -------
    mbar_energies : np.array
        mbar_energies[k][l][n] is the reduced potential sampled at state k
        and evaluated at state l at iteration n.

    """
    from openmmtools.multistate import MultiStateReporter
    reporter = MultiStateReporter(nc_file_path, open_mode='r')

    # Read the mbar energies in iteration - replica - state form.
    try:
        replica_energies, _, _ = reporter.read_energies()
        replicas_state_indices = reporter.read_replica_thermodynamic_states()
    finally:
        reporter.close()

    # Create the mbar energies in state - state - iteration form.
    n_iterations, n_replicas, n_states = replica_energies.shape
    states_energies = np.empty(shape=(n_states, n_states, n_iterations), dtype=np.float64)
    for iteration in range(n_iterations):
        state_indices = replicas_state_indices[iteration]
        states_energies[state_indices,:,iteration] = replica_energies[iteration,:,:]

    return states_energies


# =============================================================================
# THERMO LENGTH ESTIMATORS
# =============================================================================

def estimate_thermo_length_from_definition(mbar_energies, protocol):
    """Estimate the thermodynamic length from the definition.

    The du/dlambda are computed through finite differences. The function
    assumes that only one lambda parameter at a time is changed.

    Parameters
    ----------
    mbar_energies : numpy.ndarray
        mbar_energies[k][l][n] is the reduced potential sampled at state k and
        evaluated at state l at iteration n.
    protocol : Dict[str, List[float]]
        protocol[parameter_name][i] is the value of the parameter at the i-th
        state.

    Returns
    -------
    thermo_length : numpy.array
        thermo_length[i] is the thermodynamic length from state 0 to
        the i-th state.
    dthermo_length : numpy.array
        thermo_length[i] is the estimated derivative of the thermodynamic
        length with respect to the path parameter variable at state i.

    """
    n_states = len(protocol['lambda_sterics'])

    def l1_norm_diff_lambda(i, j):
        """Compute the L1 norm of the difference lambda vector between states i and j."""
        lambda_i = np.array([protocol[name][i] for name in protocol])
        lambda_j = np.array([protocol[name][j] for name in protocol])
        return np.sum(np.abs(lambda_i - lambda_j))

    # Because we assume that only one lambda variable changes at a time
    # the du/dlambda is non-zero only for one component and the thermodynamic
    # tensor is reduced to a quadratic form of a 1-dimensional covariance matrix.
    # We re-parametrize the lambda path to have a single 1-dimensional variable
    # alpha(lambda) = lambda_restraints + (1-lambda_electro) + (1-lambda_sterics).
    dthermo_length = []
    alphas = [0.0]

    for state_idx in range(n_states):
        # At the end states, we use forward/backward difference
        # while at all other states we use centered difference.
        if state_idx == 0:
            prev_state = state_idx
            next_state = state_idx + 1
        elif state_idx == n_states-1:
            prev_state = state_idx - 1
            next_state = state_idx
        else:
            prev_state = state_idx - 1
            next_state = state_idx + 1

        # Compute the alpha parameter and the tensor.
        du = mbar_energies[state_idx, next_state, :] - mbar_energies[state_idx, prev_state, :]
        dalpha = l1_norm_diff_lambda(next_state, prev_state)
        dthermo_length.append(np.std(du/dalpha, ddof=1))

        # We have already added the dalpha between the last two states in the second to last cycle.
        if state_idx < n_states-1:
            alphas.append(alphas[-1] + dalpha)

    # Compute the cumulative thermo length.
    dthermo_length = np.array(dthermo_length)
    alphas = np.array(alphas)
    thermo_length = scipy.integrate.cumtrapz(dthermo_length, alphas, initial=0.0)

    return thermo_length, dthermo_length


def compute_BAR_log_likelihood(DF, w_F, w_R):
    # return - np.mean(np.log(1 + np.exp(w_F - DF))) - np.mean(np.log(1 + np.exp(w_R + DF)))
    return - np.sum(np.log(1 + np.exp(DF - w_F))) - np.sum(np.log(1 + np.exp(-DF - w_R)))


def sparsify_mbar_energies(mbar_energies, step):
    """Reformat the mbar_energies matrix to have a smaller number of states.

    Parameters
    ----------
    mbar_energies : numpy.ndarray
        mbar_energies[k][l][n] is the reduced potential sampled at state k and
        evaluated at state l at iteration n.
    step : int, optional
        How much to sparsify states. For example, if 2, only the data
        from every other state will remain in the mbar_energies matrix.

    Returns
    -------
    sparsified_mbar_energies : numpy.ndarray
        mbar_energies[k][l][n] is the reduced potential sampled at state k*step
        and evaluated at state l*step at iteration n.

    """
    n_states = mbar_energies.shape[0]
    n_iterations = mbar_energies.shape[2]

    # Take only a fraction of states for JS and std methods.
    state_indices = list(range(0, n_states, step))
    # Always analyze the last state.
    if state_indices[-1] != n_states-1:
        state_indices.append(n_states-1)

    # Reformat the mbar energy matrix.
    if n_states != len(state_indices):
        n_states = len(state_indices)
        sparsified_mbar_energies = np.empty(shape=(n_states, n_states, n_iterations))
        for sparsified_state_idx, state_idx in enumerate(state_indices):
            sparsified_mbar_energies[sparsified_state_idx] = mbar_energies[state_idx, state_indices, :]
        mbar_energies = sparsified_mbar_energies

    return mbar_energies


def estimate_thermo_length_from_JS(mbar_energies, step=1):
    """Compute the thermodynamic length using the JS estimator in Crooks 2007 based on BAR.

    Parameters
    ----------
    mbar_energies : numpy.ndarray
        mbar_energies[k][l][n] is the reduced potential sampled at state k and
        evaluated at state l at iteration n.
    step : int, optional
        How much to sparsify states. For example, if 2, only the data
        from every other state will be used to compute the thermodynamic
        length. The JS divergence is a lower bound, so in principle a
        greater step should give you a smaller thermo length, and it should
        converge at some point with smaller step (or a great number of
        intermediate states).

    """
    from pymbar import BAR

    n_iterations = mbar_energies.shape[2]

    # Check if we need to consider only a subset of states.
    if step > 1:
        mbar_energies = sparsify_mbar_energies(mbar_energies, step)
    n_states = mbar_energies.shape[0]

    # Find JS divergences between neighbor states.
    JS_divergences = np.empty(shape=n_states-1)

    for state_idx in range(n_states-1):
        next_state_idx = state_idx + 1

        # Compute work in forward and reverse direction.
        w_F = mbar_energies[state_idx, next_state_idx, :] - mbar_energies[state_idx, state_idx, :]
        w_R = mbar_energies[next_state_idx, state_idx, :] - mbar_energies[next_state_idx, next_state_idx, :]

        # Find and store BAR solution.
        DF = BAR(w_F, w_R, compute_uncertainty=False)

        # Find log-likelihood evaluated at the solution.
        log_likelihood = compute_BAR_log_likelihood(DF, w_F, w_R)

        # Compute JS Divergence according to Crooks 2007.
        JS_divergence = log_likelihood / 2 / n_iterations + np.log(2)
        JS_divergences[state_idx] = JS_divergence

    # Estimate thermo length according to Crooks 2007.
    return np.sqrt(8) * np.cumsum(np.sqrt(JS_divergences))


def estimate_thermo_length_from_std(mbar_energies, step=1):
    """Compute the thermodynamic length using the standard deviation of the instantaneous work.

    Parameters
    ----------
    mbar_energies : numpy.ndarray
        mbar_energies[k][l][n] is the reduced potential sampled at state k and
        evaluated at state l at iteration n.
    step : int, optional
        How much to sparsify states. For example, if 2, only the data
        from every other state will be used to compute the thermodynamic
        length. The JS divergence is a lower bound, so in principle a
        greater step should give you a smaller thermo length, and it should
        converge at some point with smaller step (or a great number of
        intermediate states).

    Returns
    -------
    cumulative_std_F : numpy.ndarray
        The thermodynamic length estimate computed from the standard deviation
        of the instantaneous work in the forward direction.
    cumulative_std_R : numpy.ndarray
        The thermodynamic length estimate computed from the standard deviation
        of the instantaneous work in the reverse direction.

    """
    # Check if we need to consider only a subset of states.
    if step > 1:
        mbar_energies = sparsify_mbar_energies(mbar_energies, step)
    n_states = mbar_energies.shape[0]

    # Find std between neighbor states.
    std_F = np.empty(shape=n_states-1)
    std_R = np.empty(shape=n_states-1)

    for state_idx in range(n_states-1):
        next_state_idx = state_idx + 1

        # Compute work in forward and reverse direction.
        w_F = mbar_energies[state_idx, next_state_idx, :] - mbar_energies[state_idx, state_idx, :]
        w_R = mbar_energies[next_state_idx, state_idx, :] - mbar_energies[next_state_idx, next_state_idx, :]

        # Compute also std of work.
        std_F[state_idx] = np.std(w_F, ddof=1)
        std_R[state_idx] = np.std(w_R, ddof=1)

    # Compute the estimate of the thermo length.
    cum_std_F, cum_std_R = np.cumsum(std_F), np.cumsum(std_R)
    return cum_std_F, cum_std_R


# =============================================================================
# ACCEPTANCE RATES ANALYSIS
# =============================================================================

def lower_bound_neighbor_acceptance_rates(thermodynamic_distance, log=False):
    """The lower bound found by Shenfeld et al. for the neighbor acceptance rate.

    Parameters
    ----------
    thermodynamic_distance : float
        The thermodynamic distance between two states in kT.
    log : bool, optional, default=False
        If True, the bound on the log neighbor acceptance rate is returned.
    """
    log_lower_bound = -thermodynamic_distance**2/4 - thermodynamic_distance/np.sqrt(2)
    if log:
        return log_lower_bound
    return np.exp(log_lower_bound)


def derivative_lower_bound_neighbor_acceptance_rates(thermodynamic_distance, log=False):
    """The derivative of the acceptance rate lower bound w.r.t. the thermodynamic distance.

    Parameters
    ----------
    thermodynamic_distance : float
        The thermodynamic distance between two states in kT.
    log : bool, optional, default=False
        If True, the derivative of the bound on the log neighbor acceptance rate is returned.
    """
    s = thermodynamic_distance  # Shortcut.
    if log:
        return  - (s/2 + 1/np.sqrt(2))
    return - (s/2 + 1/np.sqrt(2)) * np.exp(-s**2/4 - s/np.sqrt(2))


def upper_bound_neighbor_acceptance_rates(thermodynamic_distance, log=False):
    """The upper bound found by Shenfeld et al. for the neighbor acceptance rate.

    Parameters
    ----------
    thermodynamic_distance : float
        The thermodynamic distance between two states in kT.
    log : bool, optional, default=False
        If True, the bound on the log neighbor acceptance rate is returned.
    """
    log_upper_bound = -thermodynamic_distance**2 / 4
    if log:
        return log_upper_bound
    return np.exp(log_upper_bound)


def derivative_upper_bound_neighbor_acceptance_rates(thermodynamic_distance, log=False):
    """The derivative of the acceptance rate upper bound w.r.t. the thermodynamic distance.

    Parameters
    ----------
    thermodynamic_distance : float
        The thermodynamic distance between two states in kT.
    log : bool, optional, default=False
        If True, the derivative of the bound on the log neighbor acceptance rate is returned.
    """
    s = thermodynamic_distance  # Shortcut.
    if log:
        return -s/2
    return - s/2 * np.exp(-s**2/4)


def compute_average_neighbor_acceptance_rates(mbar_energies, step=1):
    """Compute the average neighbor acceptance rates from the MBAR energy matrix.

    Parameters
    ----------
    mbar_energies : numpy.ndarray
        mbar_energies[k][l][n] is the reduced potential sampled at state k and
        evaluated at state l at iteration n.
    step : int, optional
        How much to sparsify states. For example, if 2, the acceptance
        rates are computed between every 2 states.

    Returns
    -------
    neighbor_acceptance_rates : np.ndarray
        neighbor_acceptance_rates[i] is the mean acceptance rate
        between states i*step and i*step+step.
    """
    n_states = mbar_energies.shape[0]
    state_indices = range(0, n_states-step, step)

    neighbor_acceptance_rates = np.empty(shape=len(state_indices))
    acceptance_rates_cis = np.empty(shape=len(state_indices))

    for i, state_idx in enumerate(state_indices):
        # Compte log p accept for all iterations.
        energy_ij = mbar_energies[state_idx, state_idx+step, :]
        energy_ji = mbar_energies[state_idx+step, state_idx, :]
        energy_ii = mbar_energies[state_idx, state_idx, :]
        energy_jj = mbar_energies[state_idx+step, state_idx+step, :]
        log_p_accept = energy_ii + energy_jj - (energy_ij + energy_ji)

        # Cap log p accept to 0.0 (max probability is 1).
        log_p_accept = np.clip(log_p_accept, a_min=None, a_max=0.0)
        p_accept = np.exp(log_p_accept)
        neighbor_acceptance_rates[i] = np.mean(p_accept)
        acceptance_rates_cis[i] = 2*scipy.stats.sem(p_accept, ddof=1)

    return neighbor_acceptance_rates