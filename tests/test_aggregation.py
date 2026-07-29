import math

from src.evaluation.aggregation import aggregate_prediction_scores, true_variables


def test_failure_aware_aggregation_keeps_invalid_predictions():
    rows = [
        {"valid_pred": 1.0, "nmse": 0.2, "nmse_var": 0.3, "r2": 0.8,
         "complexity": 5.0, "var_f1": 1.0, "var_precision": 1.0,
         "var_recall": 1.0, "sym_recovery": 1.0, "sym_skeleton": 1.0},
        {"valid_pred": 0.0, "nmse": math.inf, "nmse_var": math.inf,
         "r2": -math.inf, "complexity": 0.0, "var_f1": 0.0,
         "var_precision": 0.0, "var_recall": 0.0,
         "sym_recovery": 0.0, "sym_skeleton": 0.0},
    ]
    agg = aggregate_prediction_scores(rows, failure_nmse=100.0)
    assert agg["n_total"] == 2
    assert agg["n_valid"] == 1
    assert agg["valid_rate"] == 0.5
    assert agg["valid_nmse"] == 0.2
    assert agg["penalized_nmse"] == 50.1
    assert agg["nmse"] == agg["penalized_nmse"]
    assert agg["sym_rate"] == 0.5


def test_true_variables_are_extracted_from_expression():
    assert true_variables("2*x_2-x_1", ["x_1", "x_2", "x_3"]) == ["x_1", "x_2"]
