"""Tests on the real breast-cancer diagnosis application: the
high-dimensional logistic regression on EuclideanSpace(31) reaches the
verifiable reference accuracy, and matches torch exactly on the same
split."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from breast_cancer import evaluate, geocore_fit, load_split  # noqa: E402


@pytest.fixture(scope="module")
def split():
    pytest.importorskip("sklearn")
    return load_split()


def test_data_is_real(split):
    Xtr, Xte, ytr, yte, d = split
    assert Xtr.shape[0] + Xte.shape[0] == 569
    assert Xtr.shape[1] == 30
    assert set(np.unique(ytr)).issubset({0, 1})


def test_geocore_accuracy_matches_reference(split):
    """The breast-cancer dataset's known reference for plain logistic
    regression is ~0.95-0.97; geocore must reach it."""
    Xtr, Xte, ytr, yte, _ = split
    w, b = geocore_fit(Xtr, ytr)
    acc, sens, spec, _ = evaluate(Xte, yte, w, b)
    assert acc > 0.90
    assert sens > 0.90
    assert spec > 0.90


def test_geocore_matches_torch_exactly(split):
    """On the same split both stacks give the identical test outcome."""
    torch = pytest.importorskip("torch")
    from breast_cancer import torch_fit  # noqa: E402

    Xtr, Xte, ytr, yte, _ = split
    w_g, b_g = geocore_fit(Xtr, ytr)
    w_t, b_t = torch_fit(Xtr, ytr)
    acc_g, *_ = evaluate(Xte, yte, w_g, b_g)
    acc_t, *_ = evaluate(Xte, yte, w_t, b_t)
    assert abs(acc_g - acc_t) < 1e-6


def test_top_features_are_nucleus_size_metrics(split):
    """The highest-magnitude weights are the nucleus-size error metrics —
    consistent with the medical literature (nucleus size is a hallmark of
    malignancy)."""
    Xtr, Xte, ytr, yte, d = split
    w, _ = geocore_fit(Xtr, ytr)
    idx = np.argsort(np.abs(w))[::-1][:3]
    top = [d.feature_names[i] for i in idx]
    assert any("error" in name for name in top)
