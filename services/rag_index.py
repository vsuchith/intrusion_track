import sqlite3, os, chromadb, logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import config

log_dir = "/home/msi/Desktop/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "rag_index.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger("rag_index")

def main():
    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try: client.delete_collection("events")
    except Exception: pass
    coll = client.create_collection("events", embedding_function=ef)

    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute("SELECT cam_id, global_id, t_enter_ms, t_exit_ms FROM sessions WHERE t_enter_ms IS NOT NULL").fetchall()
    logger.info(f"[rag_index] fetched {rows} sessions and captions from DB.")
    docs, ids, metas = [], [], []
    for i,(cam,gid,tin,tout) in enumerate(rows):
        docs.append(f"global_id {gid} was in {cam} from {tin} to {tout or 'ongoing'}")
        ids.append(f"sess_{i}")
        metas.append({"type":"session","cam":cam,"gid":str(gid),"t_enter_ms":str(tin),"t_exit_ms":str(tout or 0)})
    caps = conn.execute("SELECT cam_id, caption, t_ms FROM captions").fetchall()
    logger.info(f"[rag_index] fetched {caps} captions from DB.")
    off = len(ids)
    for i,(cam,cap,tms) in enumerate(caps):
        docs.append(f"{cap} (cam={cam}, t_ms={tms})")
        ids.append(f"cap_{off+i}")
        metas.append({"type":"caption","cam":cam,"t_ms":str(tms)})
    if ids: coll.add(documents=docs, metadatas=metas, ids=ids)
    logger.info(f"[rag_index] {len(ids)} docs indexed.")

if __name__ == "__main__":
    main()
