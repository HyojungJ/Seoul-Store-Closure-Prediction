# -*- coding: utf-8 -*-
"""Print the model comparison table (values from notebooks 04 and 06)."""
from decimal import Decimal, ROUND_HALF_UP


def r3(x):
    """Round half up to 3 decimals (0.7955 -> 0.796), avoiding float artifacts."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

ROWS = [
    # model, accuracy, f1, roc_auc
    ("RandomForest", 0.7021, 0.6771, 0.7911),
    ("XGBoost",      0.7002, 0.6808, 0.7809),
    ("LightGBM",     0.7013, 0.6801, 0.7830),
    ("CatBoost",     0.7127, 0.6944, 0.7955),
    ("LSTM",         0.7061, 0.6832, 0.7750),
    ("GRU",          0.6997, 0.6865, 0.7787),
]
FINAL = "CatBoost"

line = "+" + "-" * 16 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+"
print()
print(line)
print("| {:<14} | {:>10} | {:>8} | {:>9} |".format("Model", "Accuracy", "F1", "ROC-AUC"))
print(line)
for name, acc, f1, auc in ROWS:
    mark = "*" if name == FINAL else " "
    print("| {:<14} | {:>9}{} | {:>7}{} | {:>8}{} |".format(
        name, r3(acc), mark, r3(f1), mark, r3(auc), mark))
print(line)
print("* Final model (CatBoost)")
print()

# ---------------------------------------------------------------- leakage before / after
# Experiment A: same protocol as the final model above (median target, CatBoost default
# settings, SMOTE not used). seed 42 = same seed as notebook 04 (results/leakage_exp_A_median.json)
# before = M0 (same-quarter feats, random split), after = M2 (past feats, time split)
# Experiment B (top-25% target) is a separate experiment: results/leakage_exp_B_top25.json
LEAK = [
    # metric, before (M0), after (M2)
    ("F1-score", 0.7401, 0.6944),
    ("ROC-AUC",  0.8233, 0.7955),
]
line2 = "+" + "-" * 12 + "+" + "-" * 15 + "+" + "-" * 15 + "+"
print("Data leakage - Experiment A: before vs after (CatBoost, median target, no SMOTE)")
print(line2)
print("| {:<10} | {:>13} | {:>13} |".format("Metric", "Before", "After"))
print(line2)
for metric, before, after in LEAK:
    print("| {:<10} | {:>13} | {:>13} |".format(metric, r3(before), r3(after)))
print(line2)
print("After = final CatBoost above (same test rows)")
print()
