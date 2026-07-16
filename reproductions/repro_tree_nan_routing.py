"""Repro for ML-6893: skl2onnx tree converters drop sklearn's learned NaN routing.

Run under any venv with skl2onnx installed editable from the working tree:
    <venv>/bin/python repro_tree_nan_routing.py
"""

import numpy as np
import onnx
import onnxruntime as rt
import sklearn
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def print_versions():
    print("--- versions ---")
    print("numpy       ", np.__version__)
    print("scikit-learn", sklearn.__version__)
    print("onnx        ", onnx.__version__)
    print("onnxruntime ", rt.__version__)
    print("skl2onnx    ", skl2onnx.__version__)
    print("----------------")


def make_data(n=600, d=6, nan_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    X[rng.random((n, d)) < nan_frac] = np.nan
    return X, y


def max_diff_report(name, skl_proba, onnx_proba, nan_rows):
    diff = np.abs(onnx_proba - skl_proba).max(axis=1)
    nan_max = diff[nan_rows].max() if nan_rows.any() else float("nan")
    clean_max = diff[~nan_rows].max() if (~nan_rows).any() else float("nan")
    print(f"[{name}] max diff on NaN rows:   {nan_max:.6f}")
    print(f"[{name}] max diff on clean rows: {clean_max:.6f}")
    return nan_max, clean_max


def run_random_forest():
    X, y = make_data()
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0).fit(X, y)
    assert any(np.asarray(e.tree_.missing_go_to_left).any() for e in clf.estimators_), (
        "fitted forest has no non-trivial missing_go_to_left routing -- test is meaningless"
    )

    onx = convert_sklearn(
        clf,
        initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
        options={"zipmap": False},
    )
    sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    onnx_proba = sess.run(None, {"input": X})[1]
    skl_proba = clf.predict_proba(X)
    nan_rows = np.isnan(X).any(axis=1)
    return max_diff_report("RandomForestClassifier", skl_proba, onnx_proba, nan_rows)


def run_decision_tree():
    X, y = make_data(n=400, d=4, seed=1)
    clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    assert np.asarray(clf.tree_.missing_go_to_left).any(), (
        "fitted tree has no non-trivial missing_go_to_left routing -- test is meaningless"
    )

    onx = convert_sklearn(
        clf,
        initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
        options={"zipmap": False},
    )
    sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    onnx_proba = sess.run(None, {"input": X})[1]
    skl_proba = clf.predict_proba(X)
    nan_rows = np.isnan(X).any(axis=1)
    return max_diff_report("DecisionTreeClassifier", skl_proba, onnx_proba, nan_rows)


if __name__ == "__main__":
    print_versions()
    run_random_forest()
    run_decision_tree()
