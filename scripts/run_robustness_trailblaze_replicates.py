#!/usr/env/python

"""
Simply build the experiments to make trailblaze run in ten replicates.
"""

def print_subset_jobids(exp_builder, filters):
    from yank.commands.status import find_contiguous_ids

    all_status = list(exp_builder.status())
    filtered_status = {}
    for s in exp_builder.status():
        is_filtered = False
        for f in filters:
            is_filtered |= f in s.name
        if is_filtered:
            filtered_status[s.name] = s

    print('Total number of experiments:', len(all_status))
    print('Number of filtered experiments:', len(filtered_status))
    job_ids = list(sorted([v.job_id for v in filtered_status.values()]))
    print(f'Filtered job IDs with njobs={args.njobs}:', find_contiguous_ids(job_ids))


if __name__ == '__main__':

    import argparse
    from yank.experiment import ExperimentBuilder

    parser = argparse.ArgumentParser(description='Run only the first phase of each experiment with YANK.')
    parser.add_argument('--yaml', metavar='yaml', type=str, help='Path to the YAML script.')
    parser.add_argument('--jobid', metavar='jobid', type=int, help='Job ID (from 1 to njobs).')
    parser.add_argument('--njobs', metavar='njobs', type=int, help='Total number of jobs (i.e. of experiments).')
    args = parser.parse_args()

    exp_builder = ExperimentBuilder(args.yaml, job_id=args.jobid, n_jobs=args.njobs)

    # Print status if necessary.
    # print_subset_jobids(exp_builder, filters=['forward10bidirectional_T4systemT4L99Aligands13_4'])

    for experiment in exp_builder.build_experiments():
        pass
