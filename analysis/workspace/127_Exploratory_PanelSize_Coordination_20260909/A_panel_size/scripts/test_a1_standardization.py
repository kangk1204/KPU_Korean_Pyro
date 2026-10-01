import numpy as np

from a1_local_panel_size import _inner_standardize_subset, _standardize


def test_inner_standardization_fits_only_inner_training_rows():
    xtr_raw = np.array(
        [
            [1.0, 10.0],
            [2.0, 11.0],
            [3.0, 12.0],
            [40.0, 80.0],
            [400.0, 800.0],
        ]
    )
    itr = np.array([0, 1, 2])
    ite = np.array([3])
    cols = [0, 1]

    got_train, got_test = _inner_standardize_subset(xtr_raw, itr, ite, cols)
    expected_train, expected_test = _standardize(xtr_raw[np.ix_(itr, cols)], xtr_raw[np.ix_(ite, cols)])

    np.testing.assert_allclose(got_train, expected_train)
    np.testing.assert_allclose(got_test, expected_test)
    np.testing.assert_allclose(got_train.mean(axis=0), [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(got_train.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-14)


def test_inner_standardization_is_not_influenced_by_other_outer_training_rows():
    xtr_raw = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [1000.0],
        ]
    )
    itr = np.array([0, 1, 2])
    ite = np.array([3])
    cols = [0]

    before = _inner_standardize_subset(xtr_raw, itr, ite, cols)
    xtr_raw[4, 0] = -1000.0
    after = _inner_standardize_subset(xtr_raw, itr, ite, cols)

    np.testing.assert_allclose(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1])
