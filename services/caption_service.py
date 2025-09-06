import os, time, json, pika, cv2, logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.codec import decode_frame_b64
from services.db import init
import config, sqlite3

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "caption_service.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("caption_service")

def ensure_topology(ch):
    ch.exchange_declare(exchange=config.EX_FRAMES, exchange_type='direct', durable=True)
    ch.queue_declare(queue='raw_frames_sample', durable=True)
    ch.queue_bind(queue='raw_frames_sample', exchange=config.EX_FRAMES, routing_key='raw_frames')

def main():
    conn_db = sqlite3.connect(config.DB_PATH); 
    conn_db.execute("CREATE TABLE IF NOT EXISTS captions(id INTEGER PRIMARY KEY AUTOINCREMENT, cam_id TEXT, caption TEXT, t_ms INTEGER)")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    logger.info("[caption] loaded model.")
    params = pika.URLParameters(config.RABBIT_URL)
    connection = pika.BlockingConnection(params); ch = connection.channel(); ensure_topology(ch)
    last_ts = {}
    def cb(ch, method, props, body):
        msg = eval(body.decode('utf-8'))
        cam_id = msg['cam_id']; t_ms = msg['t_ms']
        if t_ms - last_ts.get(cam_id, 0) < config.CAPTION_SAMPLE_SEC*1000: ch.basic_ack(delivery_tag=method.delivery_tag); return
        last_ts[cam_id] = t_ms
        frame = decode_frame_b64(msg['frame_b64']); image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = model.generate(**processor(images=image, return_tensors="pt"), max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"[caption generated] cam={cam_id} t_ms={t_ms} caption={caption}")
        with conn_db: conn_db.execute("INSERT INTO captions(cam_id, caption, t_ms) VALUES(?,?,?)", (cam_id, caption, t_ms))
        logger.info(f"[caption] {cam_id}: {caption}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    ch.basic_qos(prefetch_count=1); ch.basic_consume(queue='raw_frames_sample', on_message_callback=cb, auto_ack=False)
    logger.info("[caption] running.")
    try: ch.start_consuming()
    except KeyboardInterrupt: pass
    finally: ch.close(); connection.close(); conn_db.close()

if __name__ == "__main__":
    main()
