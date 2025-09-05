# This script assigns cropped (from detections) ReID embeddings to tracks 

import pika, json, numpy as np, cv2, logging, traceback
import torchvision, torchreid
from torchreid.utils import FeatureExtractor
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "reid_service.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("reid_service")

# ReID model (TorchReID)
extractor = FeatureExtractor(model_name=config.REID_MODEL, model_path='osnet_x0_25_msmt17',device='cuda' if cv2.cuda.getCudaEnabledDeviceCount()>0 else 'cpu')
logger.info(f"[Feature Extractor Initialised]")
# RabbitMQ topology
def ensure_topology(ch):
    #Subscriber Queue and Exchange Declare
    ch.exchange_declare(exchange=config.EX_TRACKS, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_TRACKS_ANY, durable=True)
    ch.queue_bind(queue=config.Q_TRACKS_ANY, exchange=config.EX_TRACKS, routing_key='tracker_frames')

    # Queue and Exchange Declare for Publisher Downstream
    ch.exchange_declare(exchange=config.EX_REID, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_REID_ANY, durable=True)
    ch.queue_bind(queue=config.Q_REID_ANY, exchange=config.EX_REID, routing_key='reid_frames')

def decode_frame_b64(b64):
    import base64, numpy as np, cv2
    data = base64.b64decode(b64.encode('ascii'))
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def crop(frame, xyxy):
    x1,y1,x2,y2 = map(int, xyxy); h,w = frame.shape[:2]
    x1=max(0,min(w-1,x1)); x2=max(0,min(w-1,x2)); y1=max(0,min(h-1,y1)); y2=max(0,min(h-1,y2))
    return frame[y1:y2, x1:x2]

def on_tracks(ch, method, props, body):
    try:    
        data = json.loads(body.decode('utf-8'))
        cam_id = data["cam_id"]
        # Base64 -> bytes -> NumPy buffer -> cv2.imdecode -> BGR image.
        frame = decode_frame_b64(data["frame_b64"])
        crops, idx = [], []
        logger.info(f"[ReID Processing] with {len(data['tracks'])} tracks from {cam_id}")
        #Clips coords to image bounds and extracts the person patch from the frame.
        for a in data["tracks"]:
            c = crop(frame, a["bbox"])
            if c.size == 0: continue
            crops.append(c[:,:,::-1])  # convert BGR → RGB for the model
            idx.append(a) # keep a reference to the original track dict
        if crops:
            # Runs all crops as a batch through the TorchReID model.
            embs = extractor(crops)  # NxD
            embs = embs.detach()
            embs = embs / (embs.norm(p=2, dim=1, keepdim=True) + 1e-12)   # L2 normalize along D
            embs_np = embs.cpu().numpy().astype(np.float32)
            # L2-normalizes each embedding vector and writes it back into the same data["tracks"] elements
            # idx holds references to those dicts
            for a, e_np in zip(idx, embs_np):
                a["embedding"] = e_np.tolist()
        out = {"cam_id": cam_id, "t_ms": data["t_ms"], "frame_id": data["frame_id"],
            "frame_b64": data["frame_b64"], "tracks": data["tracks"]}
        ch.basic_publish(exchange=config.EX_REID, routing_key="reid_frames",
                        body=json.dumps(out).encode('utf-8'),
                        properties=pika.BasicProperties(delivery_mode=2))
        logger.info(f"[ReID Published] to {config.EX_REID} with {len(data['tracks'])} tracks")
    except Exception as e:
        logging.error("reid error: %s\n%s", e, traceback.format_exc())
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
    #ch.basic_qos(prefetch_count=2)
    ch.basic_consume(queue=config.Q_TRACKS_ANY, on_message_callback=on_tracks, auto_ack=False)

    logger.info("[reid] running.")
    try: ch.start_consuming()
    except KeyboardInterrupt: pass
    finally: ch.close(); conn.close()

if __name__ == "__main__":
    main()
