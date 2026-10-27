import numpy as np

from evaluation.utils import roc_auc_or_undefined


def test_single_class_auc_is_undefined():
    score = roc_auc_or_undefined([1, 1, 1], [0.1, 0.8, 0.2])
    assert np.isnan(score)


def test_two_class_auc_is_defined():
    score = roc_auc_or_undefined([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert np.isfinite(score)
    assert score > 0.5
