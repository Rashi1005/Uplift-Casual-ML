import pandas as pd
import numpy as np
import re
import joblib
from lightgbm import LGBMClassifier
from sklift.metrics import qini_auc_score

# ---------------------------------------------------------------------------
# 1. Reload data (same as before)
# ---------------------------------------------------------------------------
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
treatment_train = pd.read_csv("treatment_train.csv")["treatment"]
treatment_test = pd.read_csv("treatment_test.csv")["treatment"]
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")
y_train_target = y_train["visit"]
y_test_target = y_test["visit"]

def clean_col_name(col):
    return re.sub(r'[^A-Za-z0-9_]+', '_', col)
X_train.columns = [clean_col_name(c) for c in X_train.columns]
X_test.columns = [clean_col_name(c) for c in X_test.columns]

# ---------------------------------------------------------------------------
# 2. Build Z target (same as before), PLUS sample weights that correct
#    for the true treatment propensity p != 0.5
# ---------------------------------------------------------------------------
p = treatment_train.mean()   # true treatment share, e.g. 0.6671
print(f"True treatment propensity p = {p:.4f}")

Z_train = ((treatment_train == 1) & (y_train_target == 1)) | ((treatment_train == 0) & (y_train_target == 0))
Z_train = Z_train.astype(int)

# weight each row inversely by its group's share -- this rebalances the
# effective distribution to 50/50 so that 2*p_z - 1 becomes valid again
sample_weight = np.where(treatment_train == 1, 1 / p, 1 / (1 - p))

class_transform_model_v2 = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    random_state=42, verbosity=-1, n_jobs=-1,
)
class_transform_model_v2.fit(X_train, Z_train, sample_weight=sample_weight)

p_z_v2 = class_transform_model_v2.predict_proba(X_test)[:, 1]
uplift_class_transform_v2 = 2 * p_z_v2 - 1

print()
print("Weighted-correction result:")
print(f"  mean = {uplift_class_transform_v2.mean():.4f}")
print(f"  min  = {uplift_class_transform_v2.min():.4f}")
print(f"  max  = {uplift_class_transform_v2.max():.4f}")

# ---------------------------------------------------------------------------
# 3. Compare Qini AUC: original vs weighted-corrected
# ---------------------------------------------------------------------------
original_combined = pd.read_csv("uplift_scores_combined.csv")
y_true = original_combined["actual_visit"].values
treatment_arr = original_combined["actual_treatment"].values

qini_original = qini_auc_score(y_true, original_combined["uplift_class_transform"].values, treatment_arr)
qini_v2 = qini_auc_score(y_true, uplift_class_transform_v2, treatment_arr)

print()
print(f"Qini AUC (original, biased)      : {qini_original:+.5f}")
print(f"Qini AUC (weighted-corrected)     : {qini_v2:+.5f}")
print(f"Difference                        : {qini_v2 - qini_original:+.5f}")

joblib.dump(class_transform_model_v2, "class_transform_model_weighted.pkl")
corrected = original_combined.copy()
corrected["uplift_class_transform_weighted"] = uplift_class_transform_v2
corrected.to_csv("uplift_scores_combined_weighted.csv", index=False)
print("\nSaved: class_transform_model_weighted.pkl")
print("Saved: uplift_scores_combined_weighted.csv")