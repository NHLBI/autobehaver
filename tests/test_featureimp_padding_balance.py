import numpy as np

from autobehaver.postinf.featureimp.hhit import generate_feature_subsets_for_hhit


def test_padding_balance_is_deterministic_and_no_duplicates():
    # Choose a feature count that is NOT divisible by n_features_per_model
    # so padding reuse is required each h_hit round.
    n_features = 23
    n_features_per_model = 10
    H = 12
    base_seed = 12345

    all_feature_names = [f"f{i:03d}" for i in range(n_features)]

    def run_once():
        extra_rep = {f: 0 for f in all_feature_names}
        all_plans = []
        all_padded = []

        for h_hit in range(1, H + 1):
            seed = base_seed + h_hit * 10000
            plan = generate_feature_subsets_for_hhit(
                all_feature_names,
                n_features_per_model=n_features_per_model,
                seed=seed,
                log_every=0,
                extra_rep_count=extra_rep,
            )

            # No duplicates within any subset
            for subset in plan.feature_subsets:
                assert len(subset) == len(set(subset))

            padded = [f for f, c in plan.feature_usage_count.items() if c > 1]
            all_padded.append(sorted(padded))

            for f in padded:
                extra_rep[f] += 1

            all_plans.append(plan.feature_subsets)

        return all_plans, all_padded, extra_rep

    plans1, padded1, extra1 = run_once()
    plans2, padded2, extra2 = run_once()

    # Deterministic for fixed seeds and identical history
    assert plans1 == plans2
    assert padded1 == padded2
    assert extra1 == extra2


def test_padding_selection_prefers_low_extra_rep_counts():
    # This test checks a key property of the min-count padding policy:
    # reused (padding) features should come from the lower end of the
    # extra_rep_count distribution among eligible candidates.
    n_features = 23
    n_features_per_model = 10
    H = 8
    base_seed = 7

    all_feature_names = [f"f{i:03d}" for i in range(n_features)]

    extra_rep = {f: 0 for f in all_feature_names}

    remainder = n_features % n_features_per_model
    assert remainder != 0
    padding_count = n_features_per_model - remainder
    assert padding_count > 0

    for h_hit in range(1, H + 1):
        seed = base_seed + h_hit * 10000
        plan = generate_feature_subsets_for_hhit(
            all_feature_names,
            n_features_per_model=n_features_per_model,
            seed=seed,
            log_every=0,
            extra_rep_count=extra_rep,
        )

        last_subset = plan.feature_subsets[-1]
        padded = [f for f in last_subset if plan.feature_usage_count.get(f, 0) > 1]
        leftover = [f for f in last_subset if plan.feature_usage_count.get(f, 0) == 1]

        # Eligible padding candidates at the time of padding are all features
        # that were already used earlier in the round, which is equivalent to
        # all features except the "leftover" unused ones.
        candidates = [f for f in all_feature_names if f not in leftover]
        cand_counts = sorted(extra_rep[f] for f in candidates)

        # The padded features should all have extra_rep_count <= the
        # padding_count-th smallest candidate count.
        threshold = cand_counts[padding_count - 1]
        assert all(extra_rep[f] <= threshold for f in padded)

        for f in padded:
            extra_rep[f] += 1
