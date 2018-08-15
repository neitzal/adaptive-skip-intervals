import argparse
import random
import sys
from itertools import cycle

argparser = argparse.ArgumentParser()

argparser.add_argument('n', type=int,
                       help='Number of hyperparameter lines to produce')
args = argparser.parse_args()

ranges_by_method = {
    'mA_1step':
        {
            'delta_t_lower_bound': [1],
            'delta_t_upper_bound': [1],
            'exploration_steps': [1],
        },
    'mB_2step':
        {
            'delta_t_lower_bound': [2],
            'delta_t_upper_bound': [2],
            'exploration_steps': [1],
        },
    'mC_temporal_matching':
        {
            'delta_t_lower_bound': [1],
            'delta_t_upper_bound': [15, 18, 21, 25],
            'exploration_steps': [7500, 10000, 15000],
        },
    'mD_temporal_matching_without_exploration':
        {
            'delta_t_lower_bound': [1],
            'delta_t_upper_bound': [15, 18, 21, 25],
            'exploration_steps': [1],
        },

}

shared_ranges = {
    'optimizer': ['adam'],
    'schd_sampling_steps': [20000],
    'f_init_learning_rate': [0.001, 0.00075, 0.0005],
    'f_learning_rate_decay_rate': [0.2],
    'f_learning_rate_decay_steps': [5000, 10000, 15000],
    'f_architecture': cycle(['f_simple', 'f_strided', 'f_dilated']),
    'n_kernels': [32, 48],
    'n_trajectories_per_batch': [2],
    'z_loss_fn': ['log_loss'],
    'initializer': ['he_uniform'],
    'activation': ['relu'],
    'dataset': ['fubo']
}

methods = [
           'mA_1step',
           'mB_2step',
           'mC_temporal_matching',
           'mD_temporal_matching_without_exploration',
           ]


def values_choice(values):
    if isinstance(values, cycle):
        return next(values)
    else:
        return random.choice(values)


def dict_choice(ranges):
    return {
        key: values_choice(values) for key, values in ranges.items()
    }


if args.n % len(methods) != 0:
    raise ValueError('n must be a multiple of the number of methods. '
                     'n = {}, len(methods) = {}'.format(args.n, len(methods)))

for _ in range(args.n // len(methods)):
    shared_params = dict_choice(shared_ranges)
    for method in methods:
        individual_params = dict_choice(ranges_by_method[method])

        key_intersection = set(shared_params.keys()) & set(individual_params.keys())
        if key_intersection:
            raise ValueError('Shared and individual params cannot have the same keys. '
                             'The keys {} appeared twice!'.format(key_intersection))
        chosen_params = {**individual_params, **shared_params}
        chosen_params['method'] = method

        print(' '.join('--{} {}'.format(key, value)
                       for key, value in chosen_params.items()))

sys.stdout.flush()
