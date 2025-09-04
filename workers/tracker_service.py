import pika, json, numpy as np, cv2, traceback, logging
from boxmot import BYTETracker
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class PerCamTracker:
    def __init__(self): self.tracker = BYTETracker()
    def update(self, dets, frame): return self.tracker.update(dets, frame)  # Mx8

def ensure_topology(ch):
    ch.exchange_declare(exchange=config.EX_DETECTIONS, exchange_type='topic', durable=True)
    ch.exchange_declare(exchange=config.EX_TRACKS, exchange_type='topic', durable=True)
    ch.queue_declare(queue=config.Q_DETS_ANY, durable=True)
    ch.queue_bind(queue=config.Q_DETS_ANY, exchange=config.EX_DETECTIONS, routing_key='cam.*')

def decode_frame_b64(b64):
    import base64, numpy as np, cv2
    data = base64.b64decode(b64.encode('ascii'))
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

state = {"per_cam": {}}

def on_detections(ch, method, props, body):
    try:    
        data = json.loads(body.decode('utf-8'))
        cam_id = data["cam_id"]
        frame = decode_frame_b64(data["frame_b64"])
        dets = np.array(data["detections"], dtype=float) if data["detections"] else np.zeros((0,6), float)
        if cam_id not in state["per_cam"]: state["per_cam"][cam_id] = PerCamTracker()
        tracks = state["per_cam"][cam_id].update(dets, frame)
        annots = []
        for row in tracks:
            x1,y1,x2,y2,tid,conf,cls,ind = [float(v) for v in row.tolist()]
            annots.append({"track_id": int(tid), "bbox": [int(x1),int(y1),int(x2),int(y2)], "conf": float(conf), "cls": int(cls)})
        out = {
            "cam_id": cam_id, "t_ms": data["t_ms"], "frame_id": data["frame_id"],
            "frame_b64": data["frame_b64"], "tracks": annots
        }
        ch.basic_publish(exchange=config.EX_TRACKS, routing_key=f"cam.{cam_id}",
                        body=json.dumps(out).encode('utf-8'),
                        properties=pika.BasicProperties(delivery_mode=2))
    
    except Exception as e:
        logging.error("tracker error: %s\n%s", e, traceback.format_exc())
        try: ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception: pass
        return
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    params = pika.URLParameters(config.RABBIT_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ensure_topology(ch)
    ch.basic_qos(prefetch_count=4)
    ch.basic_consume(queue=config.Q_DETS_ANY, on_message_callback=on_detections, auto_ack=False)
    print("[tracker] running.")
    try: ch.start_consuming()
    except KeyboardInterrupt: pass
    finally: ch.close(); conn.close()

if __name__ == "__main__":
    main()
