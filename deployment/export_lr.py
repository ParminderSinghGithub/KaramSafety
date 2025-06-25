import joblib
import numpy as np
import os

MODEL_DIR = "model_files"
OUT_HEADER = "model_lr.h"

# Load model and preprocessors
model = joblib.load(os.path.join(MODEL_DIR, "model_lr.pkl"))
scaler_final = joblib.load(os.path.join(MODEL_DIR, "scaler_lr.pkl"))
scaler_agg = joblib.load(os.path.join(MODEL_DIR, "scaler_aggregated.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "selector_lr.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Extract values
weights = model.coef_
bias = model.intercept_
mean_final = scaler_final.mean_
std_final = scaler_final.scale_
mean_agg = scaler_agg.mean_
std_agg = scaler_agg.scale_
selected_indices = selector.get_support(indices=True)
num_classes = len(label_encoder.classes_)

def array_to_c(name, arr):
    if arr.ndim == 1:
        return f"float {name}[{arr.shape[0]}] = {{" + ", ".join(f"{v:.8f}" for v in arr) + "};"
    elif arr.ndim == 2:
        rows = [", ".join(f"{v:.8f}" for v in row) for row in arr]
        return f"float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n  " + ",\n  ".join("{" + r + "}" for r in rows) + "\n};"

header = f"""\
#ifndef MODEL_LR_H
#define MODEL_LR_H

#define NUM_CLASSES {num_classes}
#define NUM_SELECTED {len(selected_indices)}
#define NUM_FEATURES 47
#define AGG_FEATURES 40

{array_to_c("weights", weights)}
{array_to_c("bias", bias)}

{array_to_c("scaler_agg_mean", mean_agg)}
{array_to_c("scaler_agg_std", std_agg)}

{array_to_c("scaler_final_mean", mean_final)}
{array_to_c("scaler_final_std", std_final)}

int selected_indices[NUM_SELECTED] = {{{", ".join(map(str, selected_indices))}}};

#endif // MODEL_LR_H
"""

with open(OUT_HEADER, "w") as f:
    f.write(header)

print("Updated 'model_lr.h' written with dual scalers and selector.")
