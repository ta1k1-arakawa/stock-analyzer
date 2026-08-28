from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.v4_meta_label_mvp import (
    FEATURE_COLUMNS, FOLDS, MODEL_PARAMS, build_meta_label_model,
    check_fold_data_sufficiency, classification_metrics, generate_oof_predictions,
    make_synthetic_phase2a_candidates, make_walk_forward_fold,
    validate_candidate_samples,
)


EXPECTED_PARAMS = {
    "objective": "binary", "n_estimators": 300, "learning_rate": 0.03,
    "num_leaves": 15, "max_depth": -1, "min_child_samples": 40,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_alpha": 0.0, "reg_lambda": 1.0, "random_state": 20260803,
    "n_jobs": 1, "deterministic": True, "force_col_wise": True,
    "verbosity": -1, "class_weight": None,
}
EXPECTED_FOLDS = (
    {"fold": 1, "train_from": "2016-04-01", "train_to": "2016-12-31", "test_from": "2017-01-01", "test_to": "2017-12-31"},
    {"fold": 2, "train_from": "2016-04-01", "train_to": "2017-12-31", "test_from": "2018-01-01", "test_to": "2018-12-31"},
    {"fold": 3, "train_from": "2016-04-01", "train_to": "2018-12-31", "test_from": "2019-01-01", "test_to": "2019-12-31"},
)


class RecordingModel:
    def __init__(self, probability: float = .6):
        self.probability = probability
        self.fit_columns: list[str] | None = None

    def fit(self, x, y):
        self.fit_columns = list(x.columns)
        self.fit_count = len(y)
        return self

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.probability), np.full(len(x), self.probability)])


def candidates() -> pd.DataFrame:
    return make_synthetic_phase2a_candidates()


def test_model_params_are_exact():
    assert MODEL_PARAMS == EXPECTED_PARAMS


def test_folds_are_exact():
    assert FOLDS == EXPECTED_FOLDS


def test_model_builder_uses_fixed_params():
    assert build_meta_label_model().get_params()["n_estimators"] == 300


def test_feature_columns_only_are_passed_to_model():
    models: list[RecordingModel] = []
    def factory():
        model = RecordingModel(); models.append(model); return model
    generate_oof_predictions(candidates(), factory)
    assert len(models) == 3 and all(model.fit_columns == list(FEATURE_COLUMNS) for model in models)


def test_ticker_is_not_a_model_input():
    model = RecordingModel()
    generate_oof_predictions(candidates(), lambda: model)
    assert "ticker" not in model.fit_columns


def test_train_signal_period_is_fold_bounded():
    train, _ = make_walk_forward_fold(candidates(), FOLDS[1])
    assert train.signal_date.between("2016-04-01", "2017-12-31").all()


def test_test_signal_period_is_fold_bounded():
    _, test = make_walk_forward_fold(candidates(), FOLDS[1])
    assert test.signal_date.between("2018-01-01", "2018-12-31").all()


def test_label_confirmed_on_test_start_is_excluded_from_train():
    frame = candidates()
    row = frame.loc[frame.signal_date == pd.Timestamp("2016-12-30")].index[0]
    frame.loc[row, "LabelConfirmedDate"] = pd.Timestamp("2017-01-01")
    train, _ = make_walk_forward_fold(frame, FOLDS[0])
    assert pd.Timestamp("2016-12-30") not in set(train.signal_date)


def test_only_pre_test_confirmed_labels_enter_train():
    train, _ = make_walk_forward_fold(candidates(), FOLDS[0])
    assert (train.LabelConfirmedDate < pd.Timestamp("2017-01-01")).all()


def test_train_one_class_is_blocked():
    train, test = make_walk_forward_fold(candidates(), FOLDS[0]); train["label"] = 1
    assert "TRAIN_LABEL_NOT_TWO_CLASSES" in check_fold_data_sufficiency(train, test)["reasons"]


def test_train_positive_below_twenty_is_blocked():
    train, test = make_walk_forward_fold(candidates(), FOLDS[0]); train["label"] = 0; train.loc[:18, "label"] = 1
    assert "TRAIN_POSITIVE_LT_20" in check_fold_data_sufficiency(train, test)["reasons"]


def test_train_negative_below_twenty_is_blocked():
    train, test = make_walk_forward_fold(candidates(), FOLDS[0]); train["label"] = 1; train.loc[:18, "label"] = 0
    assert "TRAIN_NEGATIVE_LT_20" in check_fold_data_sufficiency(train, test)["reasons"]


def test_test_one_class_is_blocked():
    train, test = make_walk_forward_fold(candidates(), FOLDS[0]); test["label"] = 1
    assert "TEST_LABEL_NOT_TWO_CLASSES" in check_fold_data_sufficiency(train, test)["reasons"]


def test_oof_period_is_2017_through_2019():
    oof = generate_oof_predictions(candidates(), RecordingModel)
    assert oof.signal_date.between("2017-01-01", "2019-12-31").all()


def test_probabilities_are_bounded():
    oof = generate_oof_predictions(candidates(), RecordingModel)
    assert oof.probability.between(0, 1).all()


def test_probability_equal_to_055_is_accepted():
    oof = generate_oof_predictions(candidates(), lambda: RecordingModel(.55))
    assert oof.decision.eq("ACCEPT").all()


def test_probability_below_055_is_abstained():
    oof = generate_oof_predictions(candidates(), lambda: RecordingModel(.549999))
    assert oof.decision.eq("ABSTAIN").all()


def test_oof_row_order_is_deterministic():
    oof = generate_oof_predictions(candidates().sample(frac=1, random_state=4), RecordingModel)
    assert oof.equals(oof.sort_values(["fold", "signal_date", "ticker"], kind="mergesort").reset_index(drop=True))


def test_same_input_produces_same_oof():
    first, second = generate_oof_predictions(candidates(), RecordingModel), generate_oof_predictions(candidates(), RecordingModel)
    assert first.equals(second)


def test_post_2019_candidate_is_rejected():
    frame = candidates(); frame.loc[0, "signal_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="POST_2019"):
        validate_candidate_samples(frame)


def test_single_class_metrics_are_blocked():
    oof = pd.DataFrame({"label": [1, 1], "probability": [.6, .7], "decision": ["ACCEPT", "ACCEPT"], "realized_net_return_percent": [1., 2.]})
    assert classification_metrics(oof)["status"] == "BLOCKED"


def test_synthetic_phase2a_smoke_test_completes():
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, "scripts/run_v4_meta_label_mvp.py", "--synthetic-phase2a-smoke-test"], cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Phase 2A synthetic smoke test passed" in result.stdout
