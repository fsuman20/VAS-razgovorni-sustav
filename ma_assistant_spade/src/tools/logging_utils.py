from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(log_dir: str, name: str = "ma_assistant") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{ts}.log")

    fmt = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info("log_file=%s", log_path)
    return logger


def log_msg(logger: logging.Logger, direction: str, sender: str, to: str, metadata: Dict[str, Any], body: str) -> None:
    rec = {
        "direction": direction,  # send|recv
        "from": sender,
        "to": to,
        "metadata": metadata,
        "body": body,
    }
    logger.info(json.dumps(rec, ensure_ascii=False))
