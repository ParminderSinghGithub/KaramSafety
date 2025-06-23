import socket
import threading
import joblib
import numpy as np
from datetime import datetime
from feature_engineering import extract_all_features
import os
import csv

# ==== Load Trained Components ====
model = joblib.load("model_files/model_lr.pkl")
scaler_agg = joblib.load("model_files/scaler_aggregated.pkl")
scaler = joblib.load("model_files/scaler_lr.pkl")
selector = joblib.load("model_files/selector_lr.pkl")
label_encoder = joblib.load("model_files/label_encoder.pkl")

# ==== Configuration ====
UDP_PORTS = {
    "module1": 12344,
    "module2": 12345
}
UDP_IP = "192.168.0.102"
BATCH_SIZE = 51
FEATURE_COUNT = 10  # ax, ay, az, gx, gy, gz, mx, my, mz, p

# ==== Select Activity ====
activity_options = [
    "idle",
    "with_hook_climbing_up",
    "with_hook_descending_down",
    "without_hook_climbing_up",
    "without_hook_descending_down"
]

print("Select the activity to record/predict:")
for i, act in enumerate(activity_options):
    print(f"{i + 1}. {act}")

selected_idx = int(input("Enter choice (1-5): ")) - 1
activity = activity_options[selected_idx]
print(f"\nSelected activity: {activity}")

# ==== Setup Paths and Buffers ====
base_log_dir = "live_testing_data"
buffers = {name: [] for name in UDP_PORTS}
data_logs = {name: [] for name in UDP_PORTS}

# ==== Create log writers per module ====
loggers = {}
for name in UDP_PORTS:
    module_dir = os.path.join(base_log_dir, name)
    os.makedirs(os.path.join(module_dir, activity), exist_ok=True)

    log_path = os.path.join(module_dir, f"pred_{name}.csv")
    file_exists = os.path.exists(log_path)
    loggers[name] = open(log_path, "a")
    
    if not file_exists:
        loggers[name].write("timestamp,prediction\n")

# ==== Reusable Inference Function ====
def process_and_predict(batch, model, selector, scaler_agg, scaler_final, label_encoder):
    aggregated = np.concatenate([
        np.mean(batch, axis=0),
        np.std(batch, axis=0),
        np.min(batch, axis=0),
        np.max(batch, axis=0)
    ])  # (40,)

    aggregated_scaled = scaler_agg.transform(aggregated.reshape(1, -1))  # (1, 40)

    engineered = extract_all_features(batch)  # (51, 18)
    engineered_mean = np.mean(engineered, axis=0).reshape(1, -1)  # (1, 18)

    full_features = np.concatenate([aggregated_scaled, engineered_mean], axis=1)  # (1, 58)
    full_features = np.nan_to_num(full_features, nan=0.0, posinf=1e6, neginf=-1e6)

    selected = selector.transform(full_features)
    final_scaled = scaler.transform(selected)

    pred = model.predict(final_scaled)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label

# ==== CSV Saving ====
def save_data_csv(name, data_rows, prediction):
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    folder_path = os.path.join(base_log_dir, name, activity)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{timestamp}_{prediction}.csv")
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_rows)

# ==== Real-Time Handler ====
def handle_module(name, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, port))
    print(f"[{name}] Listening on port {port}...")

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            decoded = data.decode().strip()
            values = list(map(float, decoded.split(",")))[:FEATURE_COUNT]
        except:
            continue

        if len(values) != FEATURE_COUNT:
            print(f"[{name}] Skipping: got {len(values)} values")
            continue

        buffers[name].append(values)
        data_logs[name].append(values)
        # print(f"[{name}] Batch length: {len(buffers[name])}")  # TEMP LOG

        if len(buffers[name]) == BATCH_SIZE:
            try:
                batch = np.array(buffers[name])
                buffers[name] = []

                label = process_and_predict(batch, model, selector, scaler_agg, scaler, label_encoder)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                loggers[name].write(f"{timestamp},{label}\n")
                loggers[name].flush()

                save_data_csv(name, data_logs[name][-BATCH_SIZE:], label)

            except Exception as e:
                print(f"[{name}] Prediction error: {e}")
                continue

# ==== Launch Threads ====
threads = []
for name, port in UDP_PORTS.items():
    t = threading.Thread(target=handle_module, args=(name, port))
    t.daemon = True
    t.start()
    threads.append(t)

print("\nReal-time prediction and recording started for both modules. Press Ctrl+C to stop.\n")

try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nStopping...")
    for f in loggers.values():
        f.close()