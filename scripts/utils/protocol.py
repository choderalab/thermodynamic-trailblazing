#!/usr/env/python

"""
Utility functions for the analysis of the trailblaze protocol.
"""

import os
import textwrap

import numpy as np


def read_experiment_protocol(experiment_dir_path):
    """Return the protocol of the experiment.

    Parameters
    ----------
    experiment_dir_path : str
        The path to the directory including the experiment data.

    Returns
    -------
    protocol : Dict[str, Dict[str, List[float]]]
        protocol[phase_name][parameter_name][i] is the value of the
        Hamiltonian parameter parameter_name for phase phase_name.
    """
    import yaml
    from yank.experiment import YankLoader

    # Load the generated YAML file containing the final protocol.
    experiment_name = os.path.basename(os.path.normpath(experiment_dir_path))
    generated_yaml_file_path = os.path.join(experiment_dir_path, experiment_name + '.yaml')
    with open(generated_yaml_file_path, 'r') as f:
        yaml_script = yaml.load(f, Loader=YankLoader)

    # Isolate the protocol.
    protocol = yaml_script['protocols']
    # There should be only a single protocol.
    assert len(protocol) == 1
    protocol_name = list(protocol.keys())[0]
    protocol = protocol[protocol_name]

    # Eliminate the trailblaze options and the key 'alchemical_path' from the dict.
    for phase_name, data in protocol.items():
        protocol[phase_name] = protocol[phase_name]['alchemical_path']

    return protocol


def format_protocol(protocol, n_states_per_row=10, indent=0):
    """Pretty-print a protocol.

    Parameters
    ----------
    protocol : Dict[str, List[float]]
        protocol[parameter_name][i] is the value of parameter_name at
        state index i.
    n_states_per_row : int, optional
        The number of protocol values for the parameter for each row.
        Default is 10.
    indent : int, optional
        The number of spaces for indentation

    Return
    ------
    protocol_str : str
        The formatted protocol as a string.
    """
    protocol_str = ''
    for parameter_name, values in protocol.items():
        protocol_str += f'{parameter_name}: ['
        for i in range(0, len(values), n_states_per_row):
            if i > 0:
                protocol_str += ',\n' + ' ' * (len(parameter_name) + 3)
            protocol_str += ', '.join([str(v) for v in values[i:i+n_states_per_row]])
        protocol_str += ']\n'

    protocol_str = textwrap.indent(protocol_str, ' '*indent)
    # Remove the last '\n' character.
    return protocol_str[:-1]


def compute_average_protocol(protocols):
    """Compute average and std of the protocols as a function of the state index.

    Parameters
    ----------
    protocols : List[Dict[str, List[float]]]
        protocols[i][parameter_name][j] is the value of parameter_name at
        state index j for protocol i. The different protocols are usually
        from different runs.

    Returns
    -------
    avg_protocol : Dict[str, List[float]]
        avg_protocol[parameter_name][j] is the average value of parameter_name
        at state index j over all protocols.
    std_protocol : Dict[str, List[float]]
        std_protocol[parameter_name][j] is the standard deviation of
        the value of parameter_name at state index j over all protocols.

    """
    # Determine all parameter names. We assume all protocols have same parameters.
    parameter_names = list(protocols[0].keys())
    # Determine maximum state index. Different protocols may have different lengths.
    max_protocol_len = max([len(list(p.values())[0]) for p in protocols])

    # Initialize return values
    avg_protocol = {n: np.empty(max_protocol_len) for n in parameter_names}
    std_protocol = {n: np.empty(max_protocol_len) for n in parameter_names}

    for parameter_name in parameter_names:
        for state_idx in range(max_protocol_len):
            # Get the values of the parameter at this state for all protocols.
            state_values = []
            for p in protocols:
                try:
                    state_values.append(p[parameter_name][state_idx])
                except IndexError:
                    # This protocol is shorter than the longest protocol.
                    pass

            # Compute average and SEM.
            avg_protocol[parameter_name][state_idx] = np.mean(state_values)
            std_protocol[parameter_name][state_idx] = np.std(state_values, ddof=1)

    return avg_protocol, std_protocol



def plot_protocol(
        axes, protocol,
        std_threshold=None,
        err_bar_protocol=None,
        err_bar_multiplier=1.0,
        plot_order=None,
        **plot_kwargs
):
    """Plot the given protocol on the Axes object.

    Parameters
    ----------
    axes : List[Axes]
        A list of axes on which to plot.
    std_threshold : float, optional
        If given, the accumulated standard deviation will be plotted on the
        x-axis instead of the number of states.
    x : str, optional
        The label to give to the x axis.
    plot_order : List, optional
        Refers to which parameter is represented on top (the first one
        in the list) and which on the bottom (the last one). It must have
        the same length of axes.
    err_bar_protocol : Dict[str, List[float]]
        The error bars for each element of the ``protocol``.
    err_bar_multiplier : float
        The values in err_bar_protocol will be scaled by this factor before
        being printed. This is useful, for example, to plot confidence intervals
        starting from SEM values passed in err_bar_protocol.

    """
    if plot_order is None:
        plot_order = ['lambda_restraints', 'lambda_electrostatics', 'lambda_sterics']

    n_states = len(protocol[plot_order[0]])

    # Fix default values.
    if std_threshold is None:
        x = list(range(n_states))
    else:
        x = np.arange(0.0, n_states*std_threshold, std_threshold)

    for idx, parameter_name in enumerate(plot_order):
        ax = axes[idx]
        par_values = protocol[parameter_name]
        ax.plot(x, par_values, **plot_kwargs)

        # Plot error bar.
        if err_bar_protocol is not None:
            ax.errorbar(x=x, y=par_values,
                        yerr=err_bar_protocol[parameter_name]*err_bar_multiplier)

        # Allow extra space above and below the max/min value to make the lines visible.
        # ax.set_ylim((min(par_values)-0.01, max(par_values)+0.01))
        ax.set_ylim((-0.01, 1.01))
        ax.set_ylabel(parameter_name.replace('lambda_', ''))

    # Set x-labels and x-ticks only in last plot.
    # for ax in axes[:-1]:
    #     ax.set_xticklabels([])
    if std_threshold is None:
        axes[-1].set_xlabel('state index')
    else:
        axes[-1].set_xlabel('s $\cdot$ state index [kT]')


if __name__ == '__main__':
    import seaborn as sns
    from matplotlib import pyplot as plt

    sns.set_style('whitegrid')
    sns.set_context('talk')

    protocol = {
        'lambda_restraints':     [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'lambda_electrostatics': [1.0, 1.0, 1.0, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
        'lambda_sterics':        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 0.0]
    }
    err_bar = {
        'lambda_restraints':     np.random.rand(9) * 0.3,
        'lambda_electrostatics': np.random.rand(9) * 0.3,
        'lambda_sterics':        np.random.rand(9) * 0.3
    }
    fig, axes = plt.subplots(3, 1)
    plot_protocol(axes, protocol, err_bar_protocol=err_bar)

    plt.tight_layout(h_pad=0.0)
    plt.show()
