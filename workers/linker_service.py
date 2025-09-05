import pika, json, numpy as np, logging, traceback
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

def _to_unit(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalized float32 1D vector; handle zeros/NaNs robustly."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32, copy=False)

class Linker:
    """
    Assigns a stable global_id (gid) across cameras using embedding similarity.
    Stores recent, L2-normalized embeddings in a time-windowed gallery.
    """
    def __init__(self, sim_thr: float, window_ms: int):
        self.sim_thr = float(sim_thr)
        self.window_ms = int(window_ms)
        self.next_gid = 1
        # recent embeddings with {gid, cam_id, emb (unit vec), t_ms}
        self.gallery = deque(maxlen=5000)
        # per-camera local track -> gid mapping
        self.cam_track_gid: dict[str, dict[int, int]] = defaultdict(dict)

    @staticmethod
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        # embeddings are already unit vectors -> cosine == dot product
        return float(np.dot(a, b))

    def assign(self, cam_id: str, tid: int, emb: np.ndarray, t_ms: int) -> int:
        # 1) Fast path: if this camera's track already has a gid, reuse it
        if tid in self.cam_track_gid[cam_id]:
            gid = self.cam_track_gid[cam_id][tid]
            # still update the gallery with the latest embedding snapshot
            self.gallery.append({"gid": gid, "cam_id": cam_id, "emb": emb, "t_ms": t_ms})
            return gid

        # 2) Cross-camera match search within the time window
        best_gid, best_sim = None, -1.0
        cutoff = t_ms - self.window_ms

        # Assumes approximate time ordering by append; if not guaranteed, remove the 'break'
        for item in reversed(self.gallery):
            if item["t_ms"] < cutoff:
                break
            if item["cam_id"] == cam_id:
                continue
            s = self.cos(emb, item["emb"])
            if s > best_sim:
                best_gid, best_sim = item["gid"], s

        # 3) Decide gid
        if best_gid is not None and best_sim >= self.sim_thr:
            gid = best_gid
        else:
            gid = self.next_gid
            self.next_gid += 1

        # 4) Record mapping and gallery
        self.cam_track_gid[cam_id][tid] = gid
        self.gallery.append({"gid": gid, "cam_id": cam_id, "emb": emb, "t_ms": t_ms})
        return gid

linker = Linker(config.SIM_THRESHOLD, config.MERGE_WINDOW_MS)

def ensure_topology(ch):
    ch.exchange_declare(exchange=config.EX_REID, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_REID_ANY, durable=True)
    ch.queue_bind(queue=config.Q_REID_ANY, exchange=config.EX_REID, routing_key='reid_frames')

    ch.exchange_declare(exchange=config.EX_GLOBAL_TRACKS, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_DISPLAY, durable=True)
    ch.queue_bind(queue=config.Q_DISPLAY, exchange=config.EX_GLOBAL_TRACKS, routing_key='global_track_frames')

def on_reid(ch, method, props, body):
    try:
        data = json.loads(body.decode('utf-8'))
        cam_id = data["cam_id"]
        t_ms = int(data["t_ms"])
        tracks = data.get("tracks", [])
        logger.info(f"[Linker] processing {len(tracks)} tracks from cam={cam_id}")

        for a in tracks:
            emb_raw = a.get("embedding")
            if emb_raw is None:
                continue
            # Normalize once; store and compare as unit vector
            emb = _to_unit(emb_raw)
            if emb.size == 0 or not np.isfinite(emb).all():
                continue  # skip bad embeddings

            tid = int(a["track_id"])
            gid = linker.assign(cam_id, tid, emb, t_ms)
            a["global_id"] = int(gid)

            # Optional: trim payload to reduce bandwidth
            # a.pop("embedding", None)

        ch.basic_publish(
            exchange=config.EX_GLOBAL_TRACKS,
            routing_key="global_track_frames",
            body=json.dumps(data).encode('utf-8'),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        # ACK only after successful publish
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logging.error("linker error: %s\n%s", e, traceback.format_exc())
        try:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception:
            pass

def main():
    params = pika.ConnectionParameters(
        host='localhost',
        port=5672,
        virtual_host='/',
        credentials=pika.PlainCredentials('guest', 'guest'),
        blocked_connection_timeout=60,
        socket_timeout=60
    )
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ensure_topology(ch)
    # let the broker throttle deliveries for steadier processing
    ch.basic_qos(prefetch_count=8)
    ch.basic_consume(queue=config.Q_REID_ANY, on_message_callback=on_reid, auto_ack=False)
    logger.info("[linker] running.")
    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ch.close()
        finally:
            conn.close()

if __name__ == "__main__":
    main()