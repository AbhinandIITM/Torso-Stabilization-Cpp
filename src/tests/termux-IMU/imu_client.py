import asyncio
import websockets
import subprocess
import threading
import json
import time
import queue
import os

SERVER_URI = "ws://localhost:8001"
q = queue.Queue()

def log(msg):
    print(f"[CLIENT] {msg}", flush=True)

def sensor_reader():
    log("Starting termux-sensor (linear accel + gyro)")

    proc = subprocess.Popen(
        [
            "termux-sensor",
            "-s", "linear Acceleration,gyroscope",
            "-d", "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    buf = ""
    depth = 0

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        depth += line.count("{")
        depth -= line.count("}")
        buf += line

        if depth == 0 and buf:
            try:
                event = json.loads(buf)
                log(f"VALID EVENT keys={list(event.keys())}")
                q.put(event)
            except json.JSONDecodeError:
                log(f"Discarding non-JSON buffer: {buf[:80]}")
            buf = ""

    stderr = proc.stderr.read()
    log("termux-sensor exited")
    if stderr:
        log(f"termux-sensor stderr:\n{stderr}")

async def ws_sender():
    log(f"Connecting to WebSocket: {SERVER_URI}")
    async with websockets.connect(SERVER_URI) as ws:
        log("WebSocket CONNECTED")

        latest = {
            "linear_acceleration": None,
            "gyroscope": None,
        }

        seq = 0

        while True:
            event = await asyncio.to_thread(q.get)

            updated = False

            if "linear Acceleration" in event:
                v = event["linear Acceleration"]["values"]
                latest["linear_acceleration"] = {
                    "x": v[0],
                    "y": v[1],
                    "z": v[2],
                }
                updated = True

            if "gyroscope" in event:
                v = event["gyroscope"]["values"]
                latest["gyroscope"] = {
                    "x": v[0],
                    "y": v[1],
                    "z": v[2],
                }
                updated = True

            if not updated:
                continue

            packet = {
                "seq": seq,
                "timestamp": time.time(),
                "pid": os.getpid(),
                "linear_acceleration": latest["linear_acceleration"],
                "gyroscope": latest["gyroscope"],
            }

            log(f"SEND seq={seq}")
            await ws.send(json.dumps(packet))
            seq += 1

def main():
    log("IMU client starting")
    threading.Thread(target=sensor_reader, daemon=True).start()
    asyncio.run(ws_sender())

if __name__ == "__main__":
    main()
