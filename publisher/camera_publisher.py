import threading, cv2, pika, time, json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.codec import encode_frame_b64, now_ms
import config

def ensure_topology(ch):
    ch.exchange_declare(exchange=config.EX_FRAMES, exchange_type='topic', durable=True)
    ch.queue_declare(queue=config.Q_FRAMES_ANY, durable=True)
    ch.queue_bind(queue=config.Q_FRAMES_ANY, exchange=config.EX_FRAMES, routing_key='cam.*')
    

def publish_camera(cam_id, src):
    #Open a separate connection per thread
    target_fps = 1           # how many frames per second you want
    interval = 1.0 / target_fps  # seconds between frames
    last_time = 0
    params = pika.URLParameters(config.RABBIT_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ensure_topology(ch)

    rk = f"cam.{cam_id}"
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[publisher] ERROR: cannot open {src}")
        ch.close(); conn.close()
        return

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"[publisher] {cam_id} EOS")
                break
            now = time.time()
            if now - last_time >= interval:
                last_time = now
                msg = {
                    "cam_id": cam_id,
                    "frame_id": frame_id,
                    "t_ms": now_ms(),
                    "frame_b64": encode_frame_b64(frame, 80),
                }
                ch.basic_publish(
                    exchange=config.EX_FRAMES,
                    routing_key=rk,
                    body=json.dumps(msg).encode("utf-8"),
                    properties=pika.BasicProperties(delivery_mode=2),
                )
                
            frame_id += 1
            print(f"[Camera {cam_id}] is publishing Frame num {frame_id}")
            time.sleep(0.001)
    finally:
        cap.release()
        ch.close()
        conn.close()

def main():
    threads = []
    for cam_id, src in config.CAMERA_SOURCES.items():
        t = threading.Thread(target=publish_camera, args=(cam_id, src), daemon=True)
        t.start()
        threads.append(t)
    print("[publisher] started. Ctrl+C to exit.")
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[publisher] stopping...")

if __name__ == "__main__":
    main()
