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

# Create log writers per module
loggers = {}
for name in UDP_PORTS:
    module_dir = os.path.join(base_log_dir, name)
    os.makedirs(os.path.join(module_dir, activity), exist_ok=True)

    log_path = os.path.join(module_dir, f"pred_{name}.csv")
    loggers[name] = open(log_path, "w")
    loggers[name].write("timestamp,prediction\n")

# ==== CSV Saving ====
def save_data_csv(name, data_rows, prediction):
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    folder_path = os.path.join(base_log_dir, name, activity)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{timestamp}_{prediction}.csv")
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_rows)

    print(f"[{name}] Data written to {file_path}")

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
        except Exception as e:
            print(f"[{name}] Invalid data skipped: {decoded} | Error: {e}")
            continue

        if len(values) != FEATURE_COUNT:
            continue

        buffers[name].append(values)
        data_logs[name].append(values)

        if len(buffers[name]) == BATCH_SIZE:
            try:
                batch = np.array(buffers[name])
                buffers[name] = []

                aggregated = np.concatenate([
                    np.mean(batch, axis=0),
                    np.std(batch, axis=0),
                    np.min(batch, axis=0),
                    np.max(batch, axis=0)
                ])

                engineered = extract_all_features(batch)
                if engineered.ndim == 1:
                    engineered = engineered.reshape(1, -1)
                else:
                    engineered = engineered.mean(axis=0).reshape(1, -1)

                final_features = np.concatenate([aggregated, engineered.flatten()])
                final_features = final_features.reshape(1, -1)

                final_features = np.nan_to_num(final_features, nan=0.0, posinf=1e6, neginf=-1e6)
                selected = selector.transform(final_features)
                scaled = scaler.transform(selected)

                pred = model.predict(scaled)[0]
                label = label_encoder.inverse_transform([pred])[0]

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{name}] {timestamp} → {label}")
                loggers[name].write(f"{timestamp},{label}\n")
                loggers[name].flush()

                save_data_csv(name, data_logs[name][-BATCH_SIZE:], label)

            except Exception as e:
                print(f"[{name}] Error during prediction: {e}")

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
