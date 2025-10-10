import logging
import time

from core.config import LOG_LEVEL

class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        extras_dict = getattr(record, "__kv__", {})
        if isinstance(extras_dict, dict):
            for k, v in extras_dict.items():
                try:
                    extras.append(f"{k}={v}")
                except Exception:
                    continue
        return base + (" | " + ",".join(extras) if extras else "")

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
handlers = [
    logging.StreamHandler(),
    logging.FileHandler("botchat_honnhan.log", encoding="utf-8"),
]
for h in handlers:
    h.setFormatter(KVFormatter(LOG_FORMAT))

root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
for h in handlers:
    root_logger.addHandler(h)

app_log = logging.getLogger("app")
metrics_logger = logging.getLogger("metrics")
if not metrics_logger.handlers:
    fh = logging.FileHandler("metrics.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    metrics_logger.addHandler(fh)
    metrics_logger.setLevel(logging.INFO)

def log_step(event: str, **kv):
    kvpairs = ",".join([f"{k}={v}" for k, v in kv.items()])
    metrics_logger.info(f"ts={int(time.time())},evt={event},{kvpairs}")

def log_time(func):
    import functools
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            app_log.info(
                f"Thời gian thực thi {func.__name__}",
                extra={"__kv__": {"thoi_gian": f"{elapsed:.4f} giây"}},
            )
    return sync_wrapper