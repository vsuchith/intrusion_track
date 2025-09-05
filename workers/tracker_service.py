import pika, json, numpy as np, cv2, traceback, logging
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "tracker_service.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("tracker_service")

class PerCamTracker:
    """
    Per-camera ByteTrack wrapper with sane defaults.
    BYTETracker requires an `args` namespace; we create it here.
    """
    def __init__(self, frame_rate=30,
                 track_thresh=0.50,     # min det score to start/keep a track
                 match_thresh=0.80,     # IoU for matching detections to tracks
                 track_buffer=30,       # keep "lost" tracks this many frames
                 min_box_area=10,       # filter tiny boxes
                 mot20=False):          # special tuning for MOT20 (usually False)
        args = SimpleNamespace(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            min_box_area=min_box_area,
            mot20=mot20
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, dets, frame):
        """
        dets: np.ndarray (N, 5) -> [x1, y1, x2, y2, score] (float32 recommended)
        frame: BGR image (H, W, 3)
        returns: list[STrack] with .tlwh and .track_id
        """
        H, W = frame.shape[:2]
        return self.tracker.update(dets, [H, W], [H, W])

# RabbitMQ topology
def ensure_topology(ch):
    #Subscriber Queue and Exchange Declare
    ch.exchange_declare(exchange=config.EX_DETECTIONS, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_DETS_ANY, durable=True)
    ch.queue_bind(queue=config.Q_DETS_ANY, exchange=config.EX_DETECTIONS, routing_key='detector_frames')

    # Queue and Exchange Declare for Publisher Downstream
    ch.exchange_declare(exchange=config.EX_TRACKS, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_TRACKS_ANY, durable=True)
    ch.queue_bind(queue=config.Q_TRACKS_ANY, exchange=config.EX_TRACKS, routing_key='tracker_frames')

def decode_frame_b64(b64):
    import base64, numpy as np, cv2
    data = base64.b64decode(b64.encode('ascii'))
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


#Each camera (cam_id) gets its own BYTETracker instance.
#Those tracker instances live in the state["per_cam"] dictionary.
#When a new frame from the same camera arrives, you call .update() on the same tracker, so it remembers previous tracks.
state = {"per_cam": {}}

def on_detections(ch, method, props, body):
    try:    
        data = json.loads(body.decode('utf-8'))
        cam_id = data["cam_id"]
        frame = decode_frame_b64(data["frame_b64"])
        # Ensure dets is (N, 5) float32: [x1,y1,x2,y2,score]
        dets = np.asarray(data["detections"],dtype=np.float32) if data["detections"] else np.zeros((0,5),dtype=np.float32)
        if cam_id not in state["per_cam"]: 
            state["per_cam"][cam_id] = PerCamTracker(frame_rate=1)
        # Tracks
        tracks = state["per_cam"][cam_id].update(dets, frame)
        logger.info(f"[Tracks Detected] with detections {dets} from {cam_id}")

        # Convert STrack objects to publishing JSON schema
        annots = []
        for t in tracks:
            # t.tlwh: (x, y, w, h) floats
            x1, y1, x2, y2 = map(int, t.tlbr)
            #x1, y1, x2, y2 = x, y, x + w, y + h

            # t.track_id: integer persistent ID within this camera/session
            tid = int(t.track_id)

            # Some builds set 'score' and 'cls' on STrack; guard with getattr
            conf = float(getattr(t, "score", 0.0))
            #cls_ = int(getattr(t, "cls", -1))

            annots.append({
                "track_id": tid,
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                #"cls": cls_
            })
        out = {
            "cam_id": cam_id, "t_ms": data["t_ms"], "frame_id": data["frame_id"],
            "frame_b64": data["frame_b64"], "tracks": annots
        }
        ch.basic_publish(exchange=config.EX_TRACKS, routing_key=f"tracker_frames",
                        body=json.dumps(out).encode('utf-8'),
                        properties=pika.BasicProperties(delivery_mode=2))
    
    except Exception as e:
        logging.error("tracker error: %s\n%s", e, traceback.format_exc())
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
    #ch.basic_qos(prefetch_count=4)
    ch.basic_consume(queue=config.Q_DETS_ANY, on_message_callback=on_detections, auto_ack=False)
    logger.info("[tracker] running.")
    try: ch.start_consuming()
    except KeyboardInterrupt: pass
    finally: ch.close(); conn.close()

if __name__ == "__main__":
    main()