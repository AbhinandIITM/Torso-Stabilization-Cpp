import asyncio
import websockets
import subprocess
import threading
import json
import time
import queue

SERVER_URI = "ws://localhost:8001"

sensor_queue = queue.Queue()

def sensor_reader():
    proc = subprocess.Popen(
        ["termux-sensor", "-s", "linear Acceleration", "-d", "10"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    buffer = ""
    depth = 0

    for line in proc.stdout:
        line = line.strip()
        depth += line.count("{")
        depth -= line.count("}")
        buffer += line

        if depth == 0 and buffer:
            try:
                sensor_queue.put(json.loads(buffer))
            except json.JSONDecodeError:
                pass
            buffer = ""

async def ws_sender():
    async with websockets.connect(SERVER_URI) as ws:
        print("Connected to IMU server")

        while True:
            data = await asyncio.to_thread(sensor_queue.get)
            payload = {
                "timestamp": time.time(),
                "imu": data
            }
            await ws.send(json.dumps(payload))

def main():
    t = threading.Thread(target=sensor_reader, daemon=True)
    t.start()
    asyncio.run(ws_sender())

if __name__ == "__main__":
    main()

