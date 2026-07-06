import numpy as np
import pandas as pd

from autobehaver.postinf.featureimp.subset_weighting import SubsetWeightingConfig, compute_tau_from_permuted


def test_tau_from_multiple_permutations_pools_all_values_and_respects_subset_complete():
    # Two permutations, two subsets; one incomplete row should be ignored.
    df = pd.DataFrame(
        {
            "perm_idx": [0, 0, 1, 1, 1],
            "h_hit": [1, 1, 1, 1, 1],
            "subset_idx": [0, 1, 0, 1, 99],
            "delta_norm": [0.01, 0.02, 0.03, 0.04, 0.999],
            "subset_complete": [True, True, True, True, False],
        }
    )

    cfg = SubsetWeightingConfig(tau_quantile=0.5, gamma=2.0)
    tau, stats = compute_tau_from_permuted(df, cfg)

    expected = float(np.quantile([0.01, 0.02, 0.03, 0.04], 0.5))
    assert tau == expected
    assert stats["n_values"] == 4
