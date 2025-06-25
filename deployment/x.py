# import socket

# PORTS = [12344, 12345]
# UDP_IP = "0.0.0.0"  # Listen on all interfaces

# # Create sockets for both ports
# sockets = []
# for port in PORTS:
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind((UDP_IP, port))
#     sockets.append(sock)
#     print(f"Listening on port {port}...")

# while True:
#     for i, sock in enumerate(sockets):
#         data, addr = sock.recvfrom(1024)
#         print(f"[Module {i+1}] {data.decode().strip()}")

import socket

PORTS = [12344, 12345]
UDP_IP = "0.0.0.0"  # Listen on all interfaces

# Create sockets for both ports
sockets = []
for port in PORTS:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, port))
    sockets.append(sock)
    print(f"Listening on port {port}...")

print("\nWaiting for data...\n")

while True:
    for i, sock in enumerate(sockets):
        data, addr = sock.recvfrom(1024)
        msg = data.decode().strip()

        if msg.startswith("F["):
            print(f"[Module {i+1}] {msg}")  # Feature index and value
        elif msg.startswith("Prediction"):
            print(f"[Module {i+1}] >>> {msg}")  # Highlight prediction
        else:
            print(f"[Module {i+1}] (Raw): {msg}")
