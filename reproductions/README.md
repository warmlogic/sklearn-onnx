# Tree NaN-routing repro

This is a minimal reproduction of a bug in the decision-tree / random-forest / extra-trees converters: scikit-learn's tree-based estimators learn a per-node `missing_go_to_left` direction for routing NaN inputs (added in recent scikit-learn versions), but skl2onnx's `add_tree_to_attribute_pairs` never reads it, so the emitted ONNX model always sends NaNs down the false branch. Predictions match scikit-learn exactly on clean rows and diverge sharply on any row with a missing value.

`HistGradientBoostingClassifier`/`Regressor` are unaffected — the separate `add_tree_to_attribute_pairs_hist_gradient_boosting` function in the same file already does this correctly, which is the precedent the fix follows.

## Running it

```bash
./setup_repro_venv.sh
.venv/bin/python repro_tree_nan_routing.py
```

The script fits a `RandomForestClassifier` and a `DecisionTreeClassifier` on data with ~15% NaNs injected, converts each to ONNX, and prints the max absolute difference between scikit-learn's and ONNX Runtime's predicted probabilities, split by whether the row has a NaN.

On the buggy converter, expect something like:

```text
[RandomForestClassifier] max diff on NaN rows:   0.699077
[RandomForestClassifier] max diff on clean rows: 0.000000
[DecisionTreeClassifier] max diff on NaN rows:   0.951220
[DecisionTreeClassifier] max diff on clean rows: 0.000000
```

Clean rows match exactly; rows with a NaN don't. After the fix in `skl2onnx/common/tree_ensemble.py`, both numbers drop to 0.

## Why onnxruntime is pinned

`setup_repro_venv.sh` pins `onnxruntime==1.25.1` instead of installing whatever's newest. This is deliberate, not laziness: onnxruntime 1.26.0 shipped with its own, unrelated regression in `TreeEnsembleClassifier`'s binary-class output — it comes out roughly complemented (`p` instead of `1 - p`) on some models. We haven't dug into _why_ that happened (it's surprising a released binary would have this), we've only confirmed it's there: with onnxruntime >= 1.26 in this same venv, even clean rows with no NaNs at all diverge from scikit-learn by ~1.0, which would swamp the much smaller NaN-routing signal this repro is trying to isolate.

So the pin exists purely to keep this repro clean. If you want to see both bugs stacked on top of each other, bump onnxruntime back up:

```bash
uv pip install --python .venv/bin/python "onnxruntime==1.27.0"
.venv/bin/python repro_tree_nan_routing.py   # NaN-row diff and clean-row diff both jump by ~1.0
uv pip install --python .venv/bin/python "onnxruntime==1.25.1"   # revert
```
