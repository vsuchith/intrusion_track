import pika, json, numpy as np, logging, traceback, logging
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from collections import deque, defaultdict

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "linker_service.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("linker_service")

class Linker:
    def __init__(self, sim_thr, window_ms):
        self.sim_thr = sim_thr; self.window_ms = window_ms
        self.next_gid = 1
        self.gallery = deque(maxlen=5000)  # {gid, cam_id, emb, t_ms}
        self.cam_track_gid = defaultdict(dict)

    @staticmethod
    def cos(a,b): return float(np.dot(a,b))

    def assign(self, cam_id, tid, emb, t_ms):
        best = (None, -1.0)
        cutoff = t_ms - self.window_ms
        for item in reversed(self.gallery):
            if item["t_ms"] < cutoff: break
            if item["cam_id"] == cam_id: continue
            s = self.cos(emb, item["emb"])
            if s > best[1]: best = (item["gid"], s)
        if best[0] is not None and best[1] >= self.sim_thr:
            gid = best[0]
        else:
            gid = self.next_gid; self.next_gid += 1
        self.gallery.append({"gid": gid, "cam_id": cam_id, "emb": emb, "t_ms": t_ms})
        self.cam_track_gid[cam_id][tid] = gid
        return gid

linker = Linker(config.SIM_THRESHOLD, config.MERGE_WINDOW_MS)

def ensure_topology(ch):
    ch.exchange_declare(exchange=config.EX_REID, exchange_type='topic', durable=True)    
    ch.queue_declare(queue=config.Q_REID_ANY, durable=True)
    ch.queue_bind(queue=config.Q_REID_ANY, exchange=config.EX_REID, routing_key='reid_frames')

    ch.exchange_declare(exchange=config.EX_GLOBAL_TRACKS, exchange_type='topic', durable=True)
    ch.queue_declare(queue=config.Q_DISPLAY, durable=True)
    ch.queue_bind(queue=config.Q_DISPLAY, exchange=config.EX_GLOBAL_TRACKS, routing_key='global_track_frames')

def on_reid(ch, method, props, body):
    try:    
        data = json.loads(body.decode('utf-8'))
        cam_id = data["cam_id"]; t_ms = data["t_ms"]
        for a in data["tracks"]:
            emb = a.get("embedding")
            if emb is None: continue
            emb = np.array(emb, dtype=np.float32)
            gid = linker.assign(cam_id, a["track_id"], emb, t_ms)
            a["global_id"] = int(gid)
        ch.basic_publish(exchange=config.EX_GLOBAL_TRACKS, routing_key="global_track_frames",
                        body=json.dumps(data).encode('utf-8'),
                        properties=pika.BasicProperties(delivery_mode=2))
    except Exception as e:
        logging.error("linker error: %s\n%s", e, traceback.format_exc())
        try: ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception: pass
        return
    #ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
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
    #ch.basic_qos(prefetch_count=8)
    ch.basic_consume(queue=config.Q_REID_ANY, on_message_callback=on_reid, auto_ack=False)
    logger.info("[linker] running.")
    try: ch.start_consuming()
    except KeyboardInterrupt: pass
    finally: ch.close(); conn.close()

if __name__ == "__main__":
    main()
