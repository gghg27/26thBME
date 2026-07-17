"""Regression tests for probability aggregation."""

import numpy as np
import pandas as pd

from V9.utils.trial_aggregation import aggregate_window_probabilities


def test_window_order_does_not_change_trial_aggregation() -> None:
    probabilities = np.asarray(
        [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32
    )
    metadata = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "b"],
            "original_trial_id": [1, 1, 2, 2],
            "pseudo_trial_id": [1, 1, 3, 3],
            "trial_id": [1, 1, 8, 8],
            "emotion_label": [0, 0, 1, 1],
            "diagnosis_label": [0, 0, 1, 1],
        }
    )
    expected = aggregate_window_probabilities(probabilities, metadata).sort_values("subject_id")
    permutation = np.asarray([2, 0, 3, 1])
    actual = aggregate_window_probabilities(
        probabilities[permutation], metadata.iloc[permutation].reset_index(drop=True)
    ).sort_values("subject_id")
    np.testing.assert_allclose(expected[["prob_0", "prob_1"]], actual[["prob_0", "prob_1"]])
    np.testing.assert_array_equal(expected["prediction"], actual["prediction"])
