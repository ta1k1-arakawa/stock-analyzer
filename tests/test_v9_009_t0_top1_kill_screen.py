from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src import v9_009_t0_top1_kill_screen as t0


def _calendar(start="2017-12-20", end="2020-01-20"):
    return pd.bdate_range(start, end).tolist()


def _frame(start="2016-09-01", periods=900, offset=0.0):
    index = pd.bdate_range(start, periods=periods)
    close = 1_000.0 + offset + np.arange(periods, dtype=float) * 0.2
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 5.0,
            "Low": close - 5.0,
            "Close": close,
            "AdjClose": close,
            "Volume": np.full(periods, 200_000.0),
        },
        index=index,
    )


def _screen_rows(year_edges):
    rows = []
    for year, edge in year_edges.items():
        day = pd.Timestamp(f"{year}-06-03")
        rows.extend(
            [
                {"d0": day, "canonical_code": "0002", "target_percentile": 0.5 + edge, "ridge_score": 2.0, "lightgbm_score": 2.0},
                {"d0": day, "canonical_code": "0001", "target_percentile": 0.1, "ridge_score": 1.0, "lightgbm_score": 1.0},
            ]
        )
    return pd.DataFrame(rows)


def test_signal_grid_is_anchored_at_first_2018_jpx_day_not_2020():
    calendar = pd.bdate_range("2017-12-25", "2020-01-10")
    grid = t0.signal_grid(calendar)
    anchor = next(i for i, day in enumerate(calendar) if day >= pd.Timestamp("2018-01-01"))
    assert calendar[anchor] in grid
    assert grid[0] == calendar[anchor - 3]
    assert all((calendar.get_loc(day) - anchor) % 3 == 0 for day in grid)


def test_all_ten_features_are_causal_and_use_frozen_formulas():
    frame = _frame(periods=400)
    day = frame.index[-1]
    values = t0.feature_values(frame, day)
    expected_return_1d = frame.loc[day, "Close"] / frame.iloc[-2].Close - 1.0
    expected_return_60d = frame.loc[day, "Close"] / frame.iloc[-61].Close - 1.0
    expected_ma20 = frame.loc[day, "Close"] / frame.Close.tail(20).mean() - 1.0
    expected_high = frame.loc[day, "Close"] / frame.Close.tail(20).max() - 1.0
    assert tuple(values) == t0.FEATURE_COLUMNS
    assert values["return_1d"] == pytest.approx(expected_return_1d)
    assert values["return_5d"] == pytest.approx(frame.loc[day, "Close"] / frame.iloc[-6].Close - 1.0)
    assert values["return_20d"] == pytest.approx(frame.loc[day, "Close"] / frame.iloc[-21].Close - 1.0)
    assert values["return_60d"] == pytest.approx(expected_return_60d)
    assert values["close_to_ma20"] == pytest.approx(expected_ma20)
    assert values["close_to_ma60"] == pytest.approx(frame.loc[day, "Close"] / frame.Close.tail(60).mean() - 1.0)
    assert values["distance_from_high20"] == pytest.approx(expected_high)
    assert values["volume_dryup"] == pytest.approx(0.0)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("Close")] = 10_000.0
    assert t0.feature_values(changed, frame.index[-2])["return_1d"] == pytest.approx(
        t0.feature_values(frame, frame.index[-2])["return_1d"]
    )


def test_causal_split_normalization_changes_only_pre_action_history():
    frame = _frame(periods=400)
    day = frame.index[-1]
    action_day = frame.index[-10]
    baseline = t0.feature_values(frame, day)
    split = t0.feature_values(frame, day, {action_day: 2.0})
    assert split["volume_dryup"] != pytest.approx(baseline["volume_dryup"])


def test_d1_to_d3_price_target_and_same_d0_percentile_ties():
    calendar = pd.bdate_range("2018-01-02", "2018-01-12")
    frame = _frame(start="2016-09-01", periods=400)
    d0 = calendar[0]
    frame = frame.reindex(frame.index.union(calendar)).sort_index().ffill()
    frame.loc[calendar[1], "Close"] = 100.0
    frame.loc[calendar[3], "Close"] = 110.0
    target, d1, d3 = t0.d1_d3_target(frame, d0, calendar)
    assert (d1, d3) == (calendar[1], calendar[3])
    assert target == pytest.approx(0.10)
    universe = pd.DataFrame({"ticker": ["0001", "0002"], "market": ["x", "x"], "industry": ["a", "b"]})
    frames = {"0001": frame, "0002": frame.copy()}
    dataset = t0.build_dataset(frames, universe, pd.bdate_range("2016-09-01", "2018-03-20"))
    tied = dataset[dataset.d0 == dataset.d0.min()]
    assert tied.target_percentile.tolist() == [0.75, 0.75]


def test_monthly_training_cutoff_excludes_current_and_future_labels():
    rows = []
    for day, exit_day in [("2017-12-20", "2017-12-28"), ("2017-12-29", "2018-01-03"), ("2018-01-02", "2018-01-05")]:
        row = {"d0": pd.Timestamp(day), "canonical_code": "0001", "target_percentile": 0.5, "target_exit_date": pd.Timestamp(exit_day), "year": 2017}
        row.update({column: 0.1 for column in t0.FEATURE_COLUMNS})
        rows.append(row)
    dataset = pd.DataFrame(rows)
    calendar = pd.bdate_range("2017-12-01", "2018-02-01")
    training = t0.causal_training_rows(dataset, calendar, 2018, 1)
    assert training.d0.tolist() == [pd.Timestamp("2017-12-20")]
    assert (training.target_exit_date < t0.month_start(calendar, 2018, 1)).all()


def test_score_ties_use_canonical_code_ascending():
    rows = _screen_rows({2020: 0.2})
    rows.loc[rows.canonical_code == "0001", "target_percentile"] = 0.9
    rows.loc[:, "ridge_score"] = 1.0
    edges = t0.rank_top1_edge(rows, "ridge_score")
    assert edges.iloc[0] == pytest.approx(0.4)
    lightgbm_edges = t0.rank_top1_edge(rows, "lightgbm_score")
    assert lightgbm_edges.iloc[0] == pytest.approx(0.2)


def test_stop_boundary_requires_both_models_and_three_or_fewer_positive_years():
    rows = _screen_rows({year: 0.0 for year in t0.KILL_SCREEN_YEARS})
    assert t0.screen_top1(rows) == "STOP"
    rows = _screen_rows({year: (0.1 if year == 2020 else 0.0) for year in t0.KILL_SCREEN_YEARS})
    assert t0.screen_top1(rows) == "CONTINUE"
    rows = _screen_rows({year: (0.1 if year in (2020, 2021, 2022) else -0.1) for year in t0.KILL_SCREEN_YEARS})
    assert t0.top1_metrics(rows, "ridge_score")["positive_years"] == 3
    assert t0.screen_top1(rows) == "STOP"
    rows = _screen_rows({year: (0.1 if year in (2020, 2021, 2022, 2023) else -0.1) for year in t0.KILL_SCREEN_YEARS})
    assert t0.top1_metrics(rows, "ridge_score")["positive_years"] == 4
    assert t0.screen_top1(rows) == "CONTINUE"


def test_aggregate_zero_is_stop_and_one_model_escape_is_continue():
    rows = _screen_rows({2020: 0.1, 2021: 0.1, 2022: 0.1, 2023: -0.1, 2024: -0.1, 2025: -0.1})
    assert t0.top1_metrics(rows, "ridge_score")["aggregate"] == pytest.approx(0.0)
    assert t0.screen_top1(rows) == "STOP"
    rows.loc[(rows.d0.dt.year == 2020) & (rows.canonical_code == "0001"), "target_percentile"] = 0.9
    rows.loc[(rows.d0.dt.year == 2020) & (rows.canonical_code == "0001"), "lightgbm_score"] = 99.0
    assert t0.screen_top1(rows) == "CONTINUE"


def test_fixed_models_have_no_search_and_fit_expected_parameterizations():
    assert t0.RIDGE_PARAMS == {"alpha": 10.0, "fit_intercept": True}
    assert t0.LIGHTGBM_PARAMS["n_estimators"] == 300
    assert t0.LIGHTGBM_PARAMS["learning_rate"] == 0.02
    assert t0.LIGHTGBM_PARAMS["num_leaves"] == 7
    assert t0.LIGHTGBM_PARAMS["max_depth"] == 3
    assert t0.LIGHTGBM_PARAMS["min_child_samples"] == 100
    assert t0.LIGHTGBM_PARAMS["random_state"] == 20260823
    assert "hyperparameter" not in inspect.getsource(t0._fit_fixed_models).lower()
    dataset = pd.DataFrame({column: np.linspace(0.0, 1.0, 120) for column in t0.FEATURE_COLUMNS})
    dataset["target_percentile"] = np.linspace(0.01, 0.99, 120)
    ridge, lightgbm = t0._fit_fixed_models(dataset)
    assert isinstance(ridge[1], __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge)
    assert lightgbm.get_params()["n_estimators"] == 300


def test_incompatible_duplicate_nonfinite_and_future_read_fail_closed():
    rows = _screen_rows({year: 0.0 for year in t0.KILL_SCREEN_YEARS})
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(t0.T0DataIncompatible, match="DUPLICATE_D0_CANONICAL_CODE"):
        t0.screen_top1(duplicate)
    bad = rows.copy()
    bad.loc[0, "ridge_score"] = np.inf
    with pytest.raises(t0.T0ImplementationFailure, match="NONFINITE_MODEL_SCORE"):
        t0.screen_top1(bad)
    frame = _frame(periods=300)
    with pytest.raises(t0.T0ImplementationFailure, match="FUTURE_PRICE_ACCESS_PROHIBITED"):
        t0.read_price(frame, frame.index[-1], "Close", frame.index[-2])


def test_safe_serializer_contains_no_outcomes_tickers_or_paths():
    result = t0.make_safe_result("NO_VERDICT_DATA_INCOMPATIBLE", "a" * 40, t0.synthetic_provenance(), cache_identity=False, exact_semantics=False)
    serialized = json.dumps(result, sort_keys=True)
    assert "aggregate" not in serialized
    assert "target_percentile" not in serialized
    assert "ridge_score" not in serialized
    assert "0001" not in serialized
    assert "\\" not in serialized
    assert set(result) == {"schema_version", "task", "design_sha", "implementation_sha", "T0_RESULT", "input_provenance", "validation"}
    assert t0.validate_safe_result(copy.deepcopy(result)) == result


def test_screen_repeatability_and_no_source_network_or_cache_writer():
    rows = _screen_rows({year: 0.0 for year in t0.KILL_SCREEN_YEARS})
    assert t0.screen_top1(rows) == t0.screen_top1(rows.copy()) == "STOP"
    source = inspect.getsource(t0)
    assert "urllib" not in source and "requests" not in source and "urlopen" not in source
    assert ".write_text" not in source and ".write_bytes" not in source and "to_csv" not in source


def _scoreable_fixture():
    calendar = pd.bdate_range("2016-09-01", "2020-01-20")
    frame = _frame(periods=900)
    universe = pd.DataFrame({"ticker": ["0001", "0002"]})
    frames = {"0001": frame.copy(), "0002": frame.copy()}
    return calendar, universe, frames


def test_d0_scoreable_population_precedes_target_attachment_and_keeps_missing_d3():
    calendar, universe, frames = _scoreable_fixture()
    grid = t0.signal_grid(calendar)
    d0 = next(day for day in grid if day.year == 2020)
    d3 = calendar[calendar.get_loc(d0) + 3]
    frames["0001"] = frames["0001"].drop(index=d3)

    population = t0.build_scoreable_population(frames, universe, calendar)
    attached = t0.attach_historical_targets(population, frames, calendar)
    dataset = t0.build_dataset(frames, universe, calendar)

    row = attached[(attached.d0 == d0) & (attached.canonical_code == "0001")].iloc[0]
    dataset_row = dataset[(dataset.d0 == d0) & (dataset.canonical_code == "0001")].iloc[0]
    assert bool(row.target_available) is False
    assert bool(dataset_row.target_available) is False
    assert ((population.d0 == d0) & (population.canonical_code == "0001")).any()


def test_removing_future_d3_does_not_change_d0_scoreability_or_model_inputs():
    calendar, universe, frames = _scoreable_fixture()
    grid = t0.signal_grid(calendar)
    d0 = next(day for day in grid if day.year == 2020)
    d3 = calendar[calendar.get_loc(d0) + 3]
    complete = t0.build_scoreable_population(frames, universe, calendar)
    frames_without_future = {code: frame.copy() for code, frame in frames.items()}
    frames_without_future["0001"] = frames_without_future["0001"].drop(index=d3)
    missing = t0.build_scoreable_population(frames_without_future, universe, calendar)

    pd.testing.assert_frame_equal(
        complete[complete.d0 <= d0].reset_index(drop=True),
        missing[missing.d0 <= d0].reset_index(drop=True),
    )
    frames_with_unusable_future_value = {code: frame.copy() for code, frame in frames.items()}
    future_day = calendar[calendar.get_loc(d0) + 1]
    frames_with_unusable_future_value["0001"].loc[future_day, "Close"] = np.inf
    nonfinite_future = t0.build_scoreable_population(
        frames_with_unusable_future_value,
        universe,
        calendar[calendar <= d0],
    )
    pd.testing.assert_frame_equal(
        complete[complete.d0 <= d0].reset_index(drop=True),
        nonfinite_future[nonfinite_future.d0 <= d0].reset_index(drop=True),
    )


def test_formal_missing_target_fails_closed_without_rank2_substitution():
    rows = _screen_rows({year: 0.0 for year in t0.KILL_SCREEN_YEARS})
    rows["target_available"] = True
    first = (rows.d0.dt.year == 2020) & (rows.canonical_code == "0001")
    rows.loc[first, "ridge_score"] = 99.0
    rows.loc[first, "lightgbm_score"] = 99.0
    rows.loc[first, "target_available"] = False
    rows.loc[first, "target_percentile"] = np.nan

    with pytest.raises(t0.T0DataIncompatible, match="FORMAL_TARGET_UNAVAILABLE"):
        t0.screen_top1(rows)


def test_training_label_availability_does_not_change_scoreable_population():
    calendar, universe, frames = _scoreable_fixture()
    population = t0.build_scoreable_population(frames, universe, calendar)
    attached = t0.attach_historical_targets(population, frames, calendar)
    attached["target_percentile"] = attached.groupby("d0", sort=False)["target_raw_return"].transform(
        lambda values: values.rank(method="average", pct=True)
    )
    missing = attached.copy()
    missing.loc[missing.index[0], "target_available"] = False
    missing.loc[missing.index[0], "target_percentile"] = np.nan
    missing.loc[missing.index[0], "target_exit_date"] = pd.NaT

    assert len(population) == len(missing)
    training = t0.causal_training_rows(missing, calendar, 2018, 1)
    assert not ((training.d0 == missing.loc[missing.index[0], "d0"]) & (training.canonical_code == missing.loc[missing.index[0], "canonical_code"])).any()


def test_d0_scoreability_never_calls_future_target_function(monkeypatch):
    calendar, universe, frames = _scoreable_fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("D1/D2/D3 target access during D0 scoreability")

    monkeypatch.setattr(t0, "d1_d3_target", forbidden)
    population = t0.build_scoreable_population(frames, universe, calendar)
    assert not population.empty


def _raw_feature_population():
    rows = []
    for day, values in [
        (pd.Timestamp("2018-01-02"), (1.0, 3.0, 3.0)),
        (pd.Timestamp("2018-01-05"), (30.0, 10.0, 20.0)),
    ]:
        for code, value in zip(("0001", "0002", "0003"), values):
            row = {"d0": day, "canonical_code": code, "year": day.year}
            row.update({column: value for column in t0.FEATURE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["d0", "canonical_code"], kind="mergesort").reset_index(drop=True)


def test_same_d0_feature_percentiles_rank_all_ten_features_with_average_ties():
    ranked = t0._rank_scoreable_features(_raw_feature_population())
    first_day = ranked[ranked.d0 == pd.Timestamp("2018-01-02")]
    second_day = ranked[ranked.d0 == pd.Timestamp("2018-01-05")]
    for column in t0.FEATURE_COLUMNS:
        assert first_day[column].tolist() == pytest.approx([1 / 3, 5 / 6, 5 / 6])
        assert second_day[column].tolist() == pytest.approx([1.0, 1 / 3, 2 / 3])


def test_feature_percentiles_are_d0_local_and_invariant_to_order_preserving_magnitudes():
    raw = _raw_feature_population()
    changed = raw.copy()
    for column in t0.FEATURE_COLUMNS:
        changed.loc[changed.d0 == pd.Timestamp("2018-01-02"), column] = [10.0, 2000.0, 2000.0]
    pd.testing.assert_frame_equal(
        t0._rank_scoreable_features(raw).loc[:, ["d0", "canonical_code", *t0.FEATURE_COLUMNS]],
        t0._rank_scoreable_features(changed).loc[:, ["d0", "canonical_code", *t0.FEATURE_COLUMNS]],
    )


def test_target_availability_is_not_an_input_to_feature_percentile_transform():
    raw = _raw_feature_population()
    with_target_state = raw.copy()
    with_target_state["target_available"] = [False, True, True, True, True, True]
    with_target_state["target_percentile"] = np.nan
    pd.testing.assert_frame_equal(
        t0._rank_scoreable_features(raw).loc[:, ["d0", "canonical_code", *t0.FEATURE_COLUMNS]],
        t0._rank_scoreable_features(with_target_state).loc[:, ["d0", "canonical_code", *t0.FEATURE_COLUMNS]],
    )


def test_fixed_models_receive_only_percentile_ranked_feature_columns(monkeypatch):
    captured = {}

    class CaptureScaler:
        def fit_transform(self, x):
            captured["scaler_fit"] = x.copy()
            return x

    class CaptureRidge:
        def __init__(self, **params):
            captured["ridge_params"] = params

        def fit(self, x, y):
            captured["ridge_fit"] = x.copy()
            return self

    class CaptureLightGBM:
        def __init__(self, **params):
            captured["lightgbm_params"] = params

        def fit(self, x, y):
            captured["lightgbm_fit"] = x.copy()
            return self

    monkeypatch.setattr(t0, "StandardScaler", CaptureScaler)
    monkeypatch.setattr(t0, "Ridge", CaptureRidge)
    monkeypatch.setattr(t0, "LGBMRegressor", CaptureLightGBM)
    train = _raw_feature_population().iloc[:3].copy()
    train.loc[:, t0.FEATURE_COLUMNS] = t0._rank_scoreable_features(train).loc[:, t0.FEATURE_COLUMNS]
    train["target_percentile"] = [1 / 3, 5 / 6, 5 / 6]

    t0._fit_fixed_models(train)
    expected = train.loc[:, t0.FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.array_equal(captured["scaler_fit"], expected)
    assert np.array_equal(captured["lightgbm_fit"], expected)
    assert np.array_equal(captured["ridge_fit"], expected)
    assert np.isfinite(expected).all() and (expected >= 0).all() and (expected <= 1).all()
