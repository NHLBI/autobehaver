"""Pool event moments and matching observed exposure across a fly's recordings."""

import numpy as np


def pool_bout_summaries(frame):
    keys = ['animal_id', 'group', 'sex', 'category', 'bout_type']
    additive = ['n_bouts', 'total_length_frames', 'n_switches', 'observed_frames',
                'observed_seconds', 'duration_sum_s', 'duration_sumsq_s',
                'interval_count', 'interval_sum_s', 'interval_sumsq_s']
    required = set(additive) - {'n_switches'}
    if not required.issubset(frame.columns):
        raise ValueError('Per-fly pooling requires fresh recording summaries with event moments.')
    result = frame.groupby(keys, observed=True)[[c for c in additive if c in frame]].sum(min_count=1).reset_index()
    count = result['n_bouts'].replace(0, np.nan)
    mean = result['duration_sum_s'] / count
    variance = (result['duration_sumsq_s'] / count - mean ** 2).clip(lower=0)
    result['bout_duration_mean_s'] = mean
    result['bout_duration_coeff_of_var'] = np.sqrt(variance) / mean.replace(0, np.nan)
    result['bout_frequency'] = result['n_bouts'] / result['observed_seconds'].replace(0, np.nan)
    available = result['observed_frames'] - result['total_length_frames']
    result['initiation_prob'] = result['n_bouts'] / available.replace(0, np.nan)
    n_intervals = result['interval_count'].replace(0, np.nan)
    interval_mean = result['interval_sum_s'] / n_intervals
    result['inter_bout_interval_mean_s'] = interval_mean
    result['inter_bout_interval_std_s'] = np.sqrt(
        (result['interval_sumsq_s'] / n_intervals - interval_mean ** 2).clip(lower=0))
    result.drop(columns=['duration_sum_s', 'duration_sumsq_s', 'interval_count',
                         'interval_sum_s', 'interval_sumsq_s', 'observed_frames'], inplace=True)
    return result.rename(columns={'animal_id': 'individual'})


def pool_sleep_summaries(frame):
    data = frame.copy()
    count = data['n_sleep_bouts']
    mean = data['sleep_bout_duration_mean_s'].fillna(0)
    sd = data['sleep_bout_duration_std_s'].fillna(0)
    data['duration_sum_s'] = count * mean
    data['duration_sumsq_s'] = count * (sd ** 2 + mean ** 2)
    result = data.groupby(['animal_id', 'group', 'sex', 'bout_type'], observed=True)[
        ['n_sleep_bouts', 'duration_sum_s', 'duration_sumsq_s',
         'sleep_total_length_frames', 'observed_frames']].sum(min_count=1).reset_index()
    n = result['n_sleep_bouts'].replace(0, np.nan)
    mean = result['duration_sum_s'] / n
    result['sleep_bout_duration_mean_s'] = mean
    result['sleep_bout_duration_std_s'] = np.sqrt(
        (result['duration_sumsq_s'] / n - mean ** 2).clip(lower=0))
    result['sleep_time_budget'] = result['sleep_total_length_frames'] / result['observed_frames'].replace(0, np.nan)
    return result.drop(columns=['duration_sum_s', 'duration_sumsq_s', 'sleep_total_length_frames',
                                'observed_frames']).rename(columns={'animal_id': 'individual'})
