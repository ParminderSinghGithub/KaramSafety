import socket
import threading
import joblib
import numpy as np
from datetime import datetime
from feature_engineering import extract_all_features
import os

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

# ==== Create log folders ====

activity = "idle"
# activity = "with_hook_climbing_up"
# activity = "with_hook_desceding_down"
# activity = "without_hook_climbing_up"
# activity = "without_hook_descending_down"

os.makedirs(f"live_testing_data/{activity}", exist_ok=True)

buffers = {name: [] for name in UDP_PORTS}
loggers = {
    name: open(f"live_testing_data/{activity}/pred_{name}.csv", "w")
    for name in UDP_PORTS
}
for f in loggers.values():
    f.write("timestamp,prediction\n")


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

        if len(buffers[name]) == BATCH_SIZE:
            try:
                batch = np.array(buffers[name])
                buffers[name] = []  # Reset buffer

                # == Match training aggregation ==
                aggregated = np.concatenate([
                    np.mean(batch, axis=0),
                    np.std(batch, axis=0),
                    np.min(batch, axis=0),
                    np.max(batch, axis=0)
                ])  # shape: (40,)

                engineered = extract_all_features(batch)  # shape: (1, 18)
                if engineered.ndim == 1:
                    engineered = engineered.reshape(1, -1)
                else:
                    engineered = engineered.mean(axis=0).reshape(1, -1)  # collapse across batch

                final_features = np.concatenate([aggregated, engineered.flatten()])  # shape: (58,)
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
            except Exception as e:
                print(f"[{name}] Error during prediction: {e}")


# ==== Launch Modules ====
threads = []
for name, port in UDP_PORTS.items():
    t = threading.Thread(target=handle_module, args=(name, port))
    t.daemon = True
    t.start()
    threads.append(t)

print("\nReal-time prediction started for both modules. Press Ctrl+C to stop.")

try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nStopping...")
    for f in loggers.values():
        f.close()
