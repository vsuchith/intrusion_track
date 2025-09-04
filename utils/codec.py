import base64, cv2, numpy as np, json, time

def encode_frame_b64(frame, quality=80):
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(buf.tobytes()).decode('ascii')

def decode_frame_b64(b64):
    data = base64.b64decode(b64.encode('ascii'))
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def now_ms():
    return int(time.time() * 1000)
