import sqlite3, os, time
import config

SCHEMA = '''
CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cam_id TEXT, global_id INTEGER, track_id INTEGER,
  x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
  conf REAL, t_ms INTEGER
);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cam_id TEXT, global_id INTEGER,
  t_enter_ms INTEGER, t_exit_ms INTEGER
);
CREATE TABLE IF NOT EXISTS captions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cam_id TEXT, caption TEXT, t_ms INTEGER
);
CREATE TABLE IF NOT EXISTS meta(
  k TEXT PRIMARY KEY, v TEXT
);
'''

def get_conn():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    return sqlite3.connect(config.DB_PATH, check_same_thread=False)

def init():
    conn = get_conn()
    with conn: conn.executescript(SCHEMA)
    return conn

def insert_track(conn, cam_id, gid, tid, bbox, conf, t_ms):
    x1,y1,x2,y2 = bbox
    with conn:
        conn.execute("INSERT INTO tracks(cam_id,global_id,track_id,x1,y1,x2,y2,conf,t_ms) VALUES(?,?,?,?,?,?,?,?,?)",
                     (cam_id, gid, tid, x1,y1,x2,y2, conf, t_ms))

def update_sessions(conn, present_ids_by_cam, last_seen, timeout_ms=2000):
    now_ms = int(time.time()*1000)
    # close timed-out
    for cam_id, seen_map in list(last_seen.items()):
        to_close = [gid for gid, t in list(seen_map.items()) if now_ms - t > timeout_ms and gid not in present_ids_by_cam.get(cam_id, set())]
        for gid in to_close:
            cur = conn.execute("SELECT id FROM sessions WHERE cam_id=? AND global_id=? AND t_exit_ms IS NULL ORDER BY id DESC LIMIT 1", (cam_id, gid))
            row = cur.fetchone()
            if row:
                with conn: conn.execute("UPDATE sessions SET t_exit_ms=? WHERE id=?", (last_seen[cam_id][gid], row[0]))
            del last_seen[cam_id][gid]
    # open/update active
    for cam_id, gids in present_ids_by_cam.items():
        for gid in gids:
            cur = conn.execute("SELECT id FROM sessions WHERE cam_id=? AND global_id=? AND t_exit_ms IS NULL ORDER BY id DESC LIMIT 1", (cam_id, gid))
            row = cur.fetchone()
            if row is None:
                with conn: conn.execute("INSERT INTO sessions(cam_id,global_id,t_enter_ms,t_exit_ms) VALUES(?,?,?,NULL)", (cam_id, gid, now_ms))
            last_seen.setdefault(cam_id, {})[gid] = now_ms
    return last_seen
