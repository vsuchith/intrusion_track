import pika, json, numpy as np, logging, traceback
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultralytics import YOLO
from utils.codec import decode_frame_b64
import config

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "detector_service.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("detector_service")
yolo = YOLO('yolov8n.pt')

def ensure_topology(ch):
    #Subscriber Queue and Exchange Declare
    ch.exchange_declare(exchange=config.EX_FRAMES, exchange_type='direct', durable=True)    
    ch.queue_declare(queue=config.Q_FRAMES_ANY, durable=True)
    ch.queue_bind(queue=config.Q_FRAMES_ANY, exchange=config.EX_FRAMES, routing_key='raw_frames')

    # Queue and Exchange Declare for Publisher Downstream
    ch.exchange_declare(exchange=config.EX_DETECTIONS, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_DETS_ANY, durable=True)
    ch.queue_bind(queue=config.Q_DETS_ANY, exchange=config.EX_DETECTIONS, routing_key='detector_frames')

def on_frame(ch, method, props, body):
    try:
        msg = json.loads(body.decode('utf-8'))   # JSON only
        cam_id = msg["cam_id"]
        frame_id = msg["frame_id"]
        frame = decode_frame_b64(msg["frame_b64"])

        res = yolo.predict(frame, conf=config.DETECT_CONF, iou=config.IOU_THRESH,
                           classes=[config.PERSON_CLASS], verbose=False)[0]

        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls  = res.boxes.cls.cpu().numpy()
            dets = np.concatenate([xyxy, conf[:,None], cls[:,None]], axis=1).tolist()
        
        out = {
            "cam_id": cam_id,
            "t_ms": msg["t_ms"],
            "frame_id": msg["frame_id"],
            "frame_b64": msg["frame_b64"],  # pass-through
            "detections": dets
        }
        ch.basic_publish(exchange=config.EX_DETECTIONS, routing_key=f"detector_frames",
                         body=json.dumps(out).encode('utf-8'),
                         properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE))
        logger.info(f"[Detections Published] with detections {dets} from {cam_id} with Frame number {frame_id}.")
    except Exception as e:
        logging.error("detector error: %s\n%s", e, traceback.format_exc())
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
    #ch.basic_qos(prefetch_count=32)
    ch.basic_consume(queue=config.Q_FRAMES_ANY, on_message_callback=on_frame, auto_ack=False)

    logger.info("[detector] running.")
    try: ch.start_consuming()
    except KeyboardInterrupt: pass
    finally: ch.close(); conn.close()

if __name__ == "__main__":
    main()
