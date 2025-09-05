import threading, cv2, pika, time, json, logging
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.codec import encode_frame_b64, now_ms
import config

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "camera_publisher.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("camera_publisher")

def ensure_topology(ch):
    ch.exchange_declare(exchange=config.EX_FRAMES, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_FRAMES_ANY, durable=True)
    ch.queue_bind(queue=config.Q_FRAMES_ANY, exchange=config.EX_FRAMES, routing_key='raw_frames')


def publish_camera(cam_id, src):
    #Open a separate connection per thread
    target_fps = 1.0           # how many frames per second you want
    interval = 1.0 / target_fps  # seconds between frames
    last_time = 0
    #params = pika.URLParameters(config.RABBIT_URL)
    params = pika.ConnectionParameters(
                                        host='localhost',        # RabbitMQ server hostname or IP
                                        port=5672,               # default AMQP port
                                        virtual_host='/',        # default vhost
                                        credentials=pika.PlainCredentials('guest', 'guest'),  # username & password
                                        blocked_connection_timeout=60,
                                        socket_timeout=60
                                        )
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ensure_topology(ch)

    rk = "raw_frames"
    cap = cv2.VideoCapture(src)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        logger.error("cannot open %s", src)
        ch.close(); conn.close()
        return

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error(f"[Camera {cam_id}] ERROR: cannot read frame")
                logger.info(f"[publisher] {cam_id} EOS")
                break
            
            if frame_id % 670 == 0:
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
                    properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
                )
                logger.info(f"[Camera {cam_id}] is publishing Frame num {frame_id}")
            frame_id += 1
                
            #time.sleep(0.001)
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

    logger.info("[publisher] started. Ctrl+C to exit.")
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logger.info("[publisher] stopping...")

if __name__ == "__main__":
    main()
