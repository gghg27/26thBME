"""Subject split balance and leakage tests."""

import pandas as pd

from V9.utils.splitting import stratified_subject_splits


def test_exact_diagnosis_balance_and_no_leakage() -> None:
    subjects = pd.DataFrame(
        {
            "subject_id": [f"hc{i}" for i in range(40)] + [f"dep{i}" for i in range(20)],
            "diagnosis_label": [0] * 40 + [1] * 20,
        }
    )
    lookup = dict(zip(subjects.subject_id, subjects.diagnosis_label))
    splits = stratified_subject_splits(subjects, n_splits=10, seed=42)
    assert len(splits) == 10
    for train, validation in splits:
        assert not set(train).intersection(validation)
        assert sum(lookup[item] == 0 for item in validation) == 4
        assert sum(lookup[item] == 1 for item in validation) == 2
