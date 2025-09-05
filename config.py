# RabbitMQ
RABBIT_URL = "amqp://guest:guest@localhost:5672/%2F"

# Two clips acting as cameras (edit these)
"""
CAMERA_SOURCES = {
    "cam1": "/home/msi/Desktop/cam1.mp4",
    "cam2": "/home/msi/Desktop/cam2.mp4",
}
"""
CAMERA_SOURCES = {
    "cam0": 0,   # /dev/video0
    "cam1": 2    # /dev/video2
}

# Exchanges
EX_FRAMES = "frames"            # publisher -> detector
EX_DETECTIONS = "detections"    # detector -> tracker
EX_TRACKS = "tracks"            # tracker -> reid
EX_REID = "reid"                # reid -> linker
EX_GLOBAL_TRACKS = "global_tracks" # linker -> display/logger

# Queues
Q_FRAMES_ANY = "frames_any"
Q_DETS_ANY = "detections_any"
Q_TRACKS_ANY = "tracks_any"
Q_REID_ANY = "reid_any"
Q_DISPLAY = "display_and_logger"

# Processing params
PERSON_CLASS = 0
DETECT_CONF = 0.35
IOU_THRESH = 0.5

# ReID / linking
REID_MODEL = "osnet_x0_25"
SIM_THRESHOLD = 0.48
MERGE_WINDOW_MS = 15000

# Captions (optional separate service can still consume frames)
CAPTION_SAMPLE_SEC = 5

# SQLite & Chroma
DB_PATH = "data/events.db"
CHROMA_DIR = "data/chroma"
