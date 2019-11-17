#!/usr/env/python

import logging
import os

import numpy as np
import yaml


logger = logging.getLogger(__name__)


# =============================================================================
# YANK ANALYSIS AT MULTIPLE ITERATIONS
# =============================================================================

def round_to_closest_multiple(values, divisor):
    """Find the list of numbers closest to values that are divisible by divisor.

    Parameters
    ----------
    values : numpy.ndarray of integers
        The values to "round".
    divisor : int
        All the returned values will be divisible by this.

    Returns
    -------
    divisible_values : numpy.ndarray of integers
        The "rounded" values.
    """
    divisible_values = np.empty(shape=values.shape, dtype=values.dtype)
    for i, value in enumerate(values):
        dividend = int(value / divisor)
        remainder = value % divisor
        if remainder > divisor / 2:
            divisible_values[i] = dividend*divisor + divisor
        else:
            divisible_values[i] = dividend*divisor
    return divisible_values


def get_energy_evaluations_per_iteration(experiment_dir_path):
    """Compute the number of energy evaluations necessary to run N iteration of the given script.

    Parameters
    ----------
    experiment_dir_path : str
        The path to the directory containing the experiment data.

    Returns
    -------
    energy_evaluations_per_iteration : int
    """
    from yank.experiment import YankLoader

    # Load the script
    experiment_name = os.path.basename(os.path.normpath(experiment_dir_path))
    with open(os.path.join(experiment_dir_path, experiment_name + '.yaml'), 'r') as f:
        script_dict = yaml.load(f, Loader=YankLoader)

    # Obtain number of states.
    protocol_name = script_dict['experiments']['protocol']
    protocol = script_dict['protocols'][protocol_name]
    n_states_complex = len(protocol['complex']['alchemical_path']['lambda_sterics'])
    n_states_solvent = len(protocol['solvent']['alchemical_path']['lambda_sterics'])
    # n_states_solvent = 62  # TODO: REMOVE ME: Solvent states in SAMPLing protocol.
    n_states = n_states_complex + n_states_solvent

    # Obtain the number of integration steps per iteration.
    n_steps_per_iterations = script_dict['options']['default_nsteps_per_iteration']

    # Compute the number of energy/force evaluations per iteration.
    md_energy_evaluations = n_states * n_steps_per_iterations
    # Rotation and translation for the complex phase require computing the initial and final energies.
    mc_energy_evaluations = 2 * 2 * n_states_complex
    # Technically, we compute only the changed force groups so this is an overestimation.
    energy_matrix_evaluations = n_states_complex**2 + n_states_solvent**2
    energy_evaluations_per_iteration = md_energy_evaluations + mc_energy_evaluations + energy_matrix_evaluations
    return energy_evaluations_per_iteration


def n_energy_eval_to_iteration(n_energy_eval, n_energy_eval_per_iteration=None, experiment_dir_path=None):
    """Return the iteration corresponding to the given number of energy evaluations.

    Parameters
    ----------
    n_energy_eval : Union[int, Iterable[int]]
        The number of energy/force evaluations.
    n_energy_eval_per_iteration : int, optional
        The number of energy evaluation per iteration.
    experiment_dir_path : str
        The path to the directory containing the experiment data, from
        which the number of energy evaluation per iteration can be determined.

    Returns
    -------
    iteration : Union[int, numpy.ndarray]
        The iteration that is closest to the given number of energy/force
        evaluations.
    """
    if n_energy_eval_per_iteration is None == experiment_dir_path is None:
        raise ValueError('One and only one between n_energy_eval_per_iteration '
                         'and experiment_dir_path must be given.')

    # Make sure n_energy_eval is an numpy array.
    if np.issubdtype(type(n_energy_eval), np.integer):
        n_energy_eval = np.array([n_energy_eval])

    # Determine the number of energy evaluations per iteration.
    if experiment_dir_path is not None:
        n_energy_eval_per_iteration = get_energy_evaluations_per_iteration(experiment_dir_path)

    # Compute the equivalent iterations.
    n_energy_eval_cutoffs = round_to_closest_multiple(n_energy_eval, n_energy_eval_per_iteration)
    iterations = np.array(n_energy_eval_cutoffs // n_energy_eval_per_iteration, dtype=np.int)

    if len(iterations) == 1:
        return iterations[0]
    return iterations


def get_analysis_cutoffs(experiment_dir_path, n_energy_eval_interval, n_iterations=None):
    """Compute the cutoffs at which to run the analysis based on the number of iterations performed.

    The function assumes that both complex and solvent phases were run
    for the same number of iterations.

    Parameters
    ----------
    experiment_dir_path : str
        The path to the directory containing the experiment data.
    n_energy_eval_interval : int
        Number of energy evaluations between analyzed iterations in the
        returned trajectory.
    n_iterations : int, optional
        The total number of iterations to consider. If not given, the function
        will read the number of iterations from the netcdf file.

    Returns
    -------
    n_energy_eval_cutoffs : numpy.array
        The number of energy evaluations to analyze.
    iteration_cutoffs : numpy.array
        The iteration numbers corresponding to the number of energy
        evaluations to analyze.

    """
    n_energy_eval_per_iteration = get_energy_evaluations_per_iteration(experiment_dir_path)
    # TODO: Warning! This assumes the solvent phase has the same number of iterations!
    nc_file_path = os.path.join(experiment_dir_path, 'complex.nc')

    if n_iterations is None:
        from openmmtools.multistate.multistatereporter import MultiStateReporter
        reporter = MultiStateReporter(nc_file_path, open_mode='r')
        try:
            n_iterations = reporter.read_last_iteration(last_checkpoint=False)
        finally:
            reporter.close()

    tot_n_energy_eval_per_iteration = n_energy_eval_per_iteration * n_iterations
    n_energy_eval_cutoffs = np.arange(n_energy_eval_interval, tot_n_energy_eval_per_iteration, n_energy_eval_interval)
    iteration_cutoffs = n_energy_eval_to_iteration(n_energy_eval_cutoffs, n_energy_eval_per_iteration)

    return n_energy_eval_cutoffs, iteration_cutoffs


def get_free_energy_traj(experiment_dir_path,
                         n_energy_eval_interval=None, n_energy_eval_cutoffs=None,
                         job_id=None, n_jobs=None):
    """Run analysis on a set of cutoffs.

    Parameters
    ----------
    experiment_dir_path : str
        The path to the directory containing the experiment data.
    n_energy_eval_interval : int, optional
        Number of energy evaluations between analyzed iterations in the
        returned trajectory.
    n_energy_eval_cutoffs : int, optional
        As an alternative to specifying n_energy_eval_interval, you can
        specify directly at which number of energy evaluations to run the analysis.
    job_id : int, optional
        The ID (from 0 to n_jobs-1) of the section of the trajectory
        computed when divided in njobs points of equal size.
    n_jobs : int, optional
        If given, the function will only compute 1/njobs-th of the trajectory.
        The trajectory is divided in non-contiguous sets so that each jobid
        takes more or less the same time to execute.

    Returns
    -------
    free_energy_traj : dict
        A map n_energy_eval -> (free_energy, free_energy_uncertainty[, bootstrap_distribution])
        The bootstrap distribution is added to the tuple only if MBAR is set
        to compute it.

    """
    from openmmtools.multistate.multistatereporter import MultiStateReporter
    from yank.analyze import YankReplicaExchangeAnalyzer

    if (n_energy_eval_interval is None) == (n_energy_eval_cutoffs is None):
        raise ValueError('One and only one between n_energy_eval_interval and '
                         'n_energy_eval_cutoffs should be specified.')

    Deltaf_traj = None
    dDeltaf_traj = None
    Deltaf_boots = None

    # Load the analysis signs.
    analysis_script_path = os.path.join(experiment_dir_path, 'analysis.yaml')
    with open(analysis_script_path, 'r') as f:
        analysis = yaml.load(f)

    for phase_name, sign in analysis:

        # Open the analyzer.
        nc_file_path = os.path.join(experiment_dir_path, phase_name + '.nc')
        reporter = MultiStateReporter(nc_file_path)
        analyzer = YankReplicaExchangeAnalyzer(reporter)

        # Obtain the iterations cutoffs at which to analyze the calculation.
        if n_energy_eval_cutoffs is None:
            n_energy_eval_cutoffs, iteration_cutoffs = get_analysis_cutoffs(
                experiment_dir_path, n_energy_eval_interval,
                n_iterations=analyzer.n_iterations)
        else:
            n_energy_eval_per_iteration = get_energy_evaluations_per_iteration(experiment_dir_path)
            iteration_cutoffs = n_energy_eval_to_iteration(n_energy_eval_cutoffs, n_energy_eval_per_iteration)

        # Select the iterations that need to be computed.
        if n_jobs is not None:
            iteration_cutoffs = iteration_cutoffs[job_id::n_jobs]
            n_energy_eval_cutoffs = n_energy_eval_cutoffs[job_id::n_jobs]

        # Initialize the free energy trajectory variable.
        if Deltaf_traj is None:
            Deltaf_traj = np.zeros(shape=len(iteration_cutoffs))
            dDeltaf_traj = np.zeros(shape=len(iteration_cutoffs))

        # Get the free energy for all iterations.
        for i, iteration_cutoff in enumerate(iteration_cutoffs):
            logger.debug(f'Analyzing {phase_name} at iteration {iteration_cutoff} ({i+1}/{len(iteration_cutoffs)})')

            analyzer.max_n_iterations = iteration_cutoff
            Deltaf_ij, dDeltaf_ij = analyzer.get_free_energy()
            standard_state_correction = analyzer.get_standard_state_correction()
            Deltaf_traj[i] -= sign * (Deltaf_ij[0, -1] + standard_state_correction)
            # We'll take the square root of the complex+solvent free energy uncertainty later.
            dDeltaf_traj[i] += dDeltaf_ij[0, -1]**2

            # Check if there are free energy bootstrap distributions.
            if hasattr(analyzer.mbar, 'f_k_boots'):
                if Deltaf_boots is None:
                    Deltaf_boots = np.empty(shape=(len(iteration_cutoffs), analyzer.mbar.nbootstraps))
                Deltaf_boots[i] -= sign * (analyzer.mbar.f_k_boots[:,-1] - analyzer.mbar.f_k_boots[:,0] + standard_state_correction)

        # TODO: REMOVE ME: Solvent free energies in SAMPLing.
        # solvent_free_energy = np.mean([130.339, 130.344, 130.213, 130.320])
        # Deltaf_traj += sign * solvent_free_energy
        # if Deltaf_boots is not None:
        #     Deltaf_boots += sign * solvent_free_energy

    # Take the square root of the uncertainty for error propagation.
    dDeltaf_traj = np.sqrt(dDeltaf_traj)

    # Convert the trajectory into dictionary format.
    if Deltaf_boots is not None:
        Deltaf_boots = Deltaf_boots.tolist()

    free_energy_traj = {}
    for i, n_energy_eval_cutoff in enumerate(n_energy_eval_cutoffs):
        if Deltaf_boots is None:
            free_energy_traj[n_energy_eval_cutoff] = (Deltaf_traj[i], dDeltaf_traj[i])
        else:
            free_energy_traj[n_energy_eval_cutoff] = (Deltaf_traj[i], dDeltaf_traj[i], Deltaf_boots[i])

    return free_energy_traj


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def compute_mean_free_energy_traj(free_energy_trajectories):
    """Compute the mean and std of replicate trajectories that can be of different lengths.

    Short trajectories are simply ignored when computing the mean and
    standard deviation at large numbers of energy/force evaluations.

    Parameters
    ----------
    free_energy_trajectories : List[numpy.array]
        free_energy_trajectories[i][j] is the free energy obtained by replicate i
        at the j-th analyzed number of energy evaluations.

    Returns
    -------
    mean_free_energy_traj : numpy.array
        mean_free_energy_traj[j] is the mean free energy at the j-th analyzed number
        of energy evaluations.
    std_free_energy_traj : numpy.array, optional
        std_free_energy_traj[j] is the standard deviation of the free energy at the
        j-th analyzed number of energy evaluations.
    """
    # If the free energy trajectories all have the same lengths simply return the mean.
    all_lengths = {len(fe) for fe in free_energy_trajectories}
    if len(all_lengths) == 1:
        f_mean = np.mean(free_energy_trajectories, axis=0)
        f_std = np.std(free_energy_trajectories, axis=0, ddof=1)
        return f_mean, f_std

    # Otherwise, create a masked array so that the mean can be computed normally.
    max_len = max(all_lengths)
    n_replicates = len(free_energy_trajectories)
    masked_fe = np.ma.empty((n_replicates, max_len))
    masked_fe.mask = True

    # Build the masked array.
    for i, fe in enumerate(free_energy_trajectories):
        masked_fe[i,:len(fe)] = fe

    # Compute the mean and std.
    f_mean = masked_fe.mean(axis=0)
    f_std = masked_fe.std(axis=0, ddof=1)

    # Remove the mask before returning.
    return f_mean.data, f_std.data
