import pika, json, cv2, logging, os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from utils.codec import decode_frame_b64
from services.db import init, insert_track, update_sessions
import numpy as np

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "display_service.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("display_service")

def ensure_topology(ch):
    ch.exchange_declare(exchange=config.EX_GLOBAL_TRACKS, exchange_type='direct', durable=True)
    ch.queue_declare(queue=config.Q_DISPLAY, durable=True)
    ch.queue_bind(queue=config.Q_DISPLAY, exchange=config.EX_GLOBAL_TRACKS, routing_key='global_track_frames')

"""
Example message body (JSON):
{
  "cam_id": "camA",
  "t_ms": 10000,
  "tracks": [
    {"bbox":[100,120,220,380], "track_id": 4, "global_id": 7, "conf":0.92},
    {"bbox":[300,150,420,360], "track_id": 1, "global_id": 12, "conf":0.88}
  ]
}

-> insert_track writes 2 rows to tracks table
-> present_ids_by_cam = {"camA": {7, 12}}
-> update_sessions finds no open session for (camA,7) or (camA,12), so it inserts two sessions rows with t_enter_ms = now_ms. 
-> It then sets last_seen["camA"][7] = now_ms, 
   last_seen["camA"][12] = now_ms

Next message at t_ms=11000 shows only global_id=7:

-> present_ids_by_cam = {"camA": {7}}
-> update_sessions: 
    -> For 12, if now_ms - last_seen["camA"][12] > timeout_ms it will close its session by writing t_exit_ms
    -> It keeps 7 open and refreshes last_seen["camA"][7]

If the very next message is for camB at t_ms=11200 and includes global_id=7,
update_sessions will open a session for (camB,7) as well. This enables later cross-camera “transition” logic.

"""

def on_msg(ch, method, props, body, state):
    data = json.loads(body.decode('utf-8'))
    cam_id = data["cam_id"]
    t_ms = data["t_ms"]
    frame = decode_frame_b64(data["frame_b64"])
    logger.info(f"[display and logger service] got {len(data.get('tracks', []))} tracks from cam={cam_id}")
    present = set()
    for a in data.get("tracks", []):
        gid = int(a.get("global_id", -1))
        if gid < 0: continue
        tid = int(a["track_id"])
        x1,y1,x2,y2 = map(int, a["bbox"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"G{gid}/T{tid}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        insert_track(state["conn"], cam_id, gid, tid, (x1,y1,x2,y2), float(a.get("conf",1.0)), t_ms)
        logger.info(f"[db] inserted track cam={cam_id} gid={gid} tid={tid} bbox=({x1},{y1},{x2},{y2}) conf={a.get('conf',1.0)} t_ms={t_ms}")
        present.add(gid)
    # {"camA": {7, 12}}
    logger.info(f"[sessions] updating sessions with present ids by cam: {{{cam_id}: {present}}}")
    state["last_seen"] = update_sessions(state["conn"], {cam_id: present}, state["last_seen"], now_ms=t_ms)
    logger.info(f"[sessions] updated last_seen: {state['last_seen']}")

    # --- window management: one window per cam_id ---
    if "windows" not in state:
        state["windows"] = {}

    if cam_id not in state["windows"]:
        # Create a resizable window and tile it
        cv2.namedWindow(cam_id, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam_id, 960, 540)  # adjust to taste

        # simple tiling: place based on count
        idx = len(state["windows"])
        x_offset = (idx % 2) * 980
        y_offset = (idx // 2) * 560
        try:
            cv2.moveWindow(cam_id, x_offset, y_offset)
        except Exception:
            pass

        state["windows"][cam_id] = True

    cv2.imshow(cam_id, frame)

    # ESC closes all
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        ch.stop_consuming()

    # If user closes a window manually (click X), stop consuming
    try:
        if cv2.getWindowProperty(cam_id, cv2.WND_PROP_VISIBLE) < 1:
            ch.stop_consuming()
    except Exception:
        pass    


    #cv2.imshow(cam_id, frame)
    #if cv2.waitKey(1) & 0xFF == 27:
    #    ch.stop_consuming()
    #ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    conn = init()
    state = {"conn": conn, "last_seen": {}}
    #params = pika.URLParameters(config.RABBIT_URL)
    params = pika.ConnectionParameters(
                                        host='localhost',        # RabbitMQ server hostname or IP
                                        port=5672,               # default AMQP port
                                        virtual_host='/',        # default vhost
                                        credentials=pika.PlainCredentials('guest', 'guest'),  # username & password
                                        blocked_connection_timeout=60,
                                        socket_timeout=60
                                        )
    connection = pika.BlockingConnection(params)
    ch = connection.channel(); ensure_topology(ch)
    #ch.basic_qos(prefetch_count=10)
    ch.basic_consume(queue=config.Q_DISPLAY, on_message_callback=lambda ch,m,p,b: on_msg(ch,m,p,b,state), auto_ack=False)
    print("[display] running. ESC to close.")
    try: ch.start_consuming()
    except KeyboardInterrupt: pass
    finally: ch.close(); connection.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
