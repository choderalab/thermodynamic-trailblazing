#!/usr/env/python

"""
Run YANK experiments for a number of iterations necessary to reach a certain number of energy evaluations.
"""

import math
import os


# Total number of energy evaluations that we want to run.
N_ENERGY_EVALS = {'CB8': 7e8, 'T4': 2e8}


# TODO: This function was copied from scripts/utils/analysis.py
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
    import yaml
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


if __name__ == '__main__':
    import argparse
    from yank.experiment import ExperimentBuilder

    parser = argparse.ArgumentParser(description='Run only the first phase of each experiment with YANK.')
    parser.add_argument('--yaml', metavar='yaml', type=str, help='Path to the YAML script.')
    parser.add_argument('--jobid', metavar='jobid', type=int, help='Job ID (from 1 to njobs).')
    parser.add_argument('--njobs', metavar='njobs', type=int, help='Total number of jobs (i.e. of experiments).')
    parser.add_argument('--build', action='store_true', help='Only build the experiments')
    parser.add_argument('--setniterations', action='store_true', help='Just set the final number of iterations without running')
    parser.add_argument('--status', action='store_true', help='Only build the experiments')
    args = parser.parse_args()

    exp_builder = ExperimentBuilder(args.yaml, job_id=args.jobid, n_jobs=args.njobs)
    for experiment in exp_builder.build_experiments():
        # Check if we need to only build the experiments.
        if args.build:
            continue

        # Find the experiment directory path.
        complex_phase = experiment.phases[0]
        if isinstance(complex_phase, str):
            # Resuming.
            experiment_dir_path = os.path.dirname(complex_phase)
        else:
            # This is an AlchemicalPhaseFactory object.
            experiment_dir_path = os.path.dirname(complex_phase.storage)

        # Find the total number of free energy calculations to run.
        n_energy_evals_per_iteration = get_energy_evaluations_per_iteration(experiment_dir_path)
        for system_name, target_n_energy_eval in N_ENERGY_EVALS.items():
            if system_name in experiment_dir_path:
                break

        # TODO: REMOVE ME - Initially, run only for half the time.
        # target_n_energy_eval /= 2

        # Truncate to the nearest checkpoint interval.
        n_iterations_to_run = int(math.ceil(target_n_energy_eval / n_energy_evals_per_iteration))
        n_iterations_to_run = int(round(n_iterations_to_run, -3))

        # Run at least 10000 iterations.
        # This is just for experiment-T4-main/trailblaze05_T4systemT4L99Aligands4: job_id=54
        # n_iterations_to_run = max(n_iterations_to_run, 10000)

        # Check if we just need to set the number of iterations.
        if args.setniterations is True:
            from yank.yank import AlchemicalPhase

            for phase in experiment.phases:
                # We need to change the number of iterations only if we're
                # resuming otherwise this will be set correctly on creation.
                # Check also that the number of iteration needs to be changed.
                if isinstance(phase, str) and AlchemicalPhase.read_status(phase) != n_iterations_to_run:
                    # Resume.
                    alchemical_phase = AlchemicalPhase.from_storage(phase)
                    alchemical_phase.number_of_iterations = n_iterations_to_run
                    del alchemical_phase

            # We don't run the experiment when setniterations is set.
            continue

        # Run experiment.
        experiment.number_of_iterations = n_iterations_to_run
        experiment.run()
