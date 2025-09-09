# logging_utils.py

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Tuple, Mapping, Any, Iterable


def setup_logger(
    name: str,
    logs_dir: str = "logs",
    level: int = logging.INFO,
    announce: bool = False,
) -> Tuple[logging.Logger, str]:
    """Create a rotating file + console logger and return (logger, log_path).

    - Daily rotation, keep 7 backups.
    - Idempotent in notebooks (won't add duplicate handlers).
    - If `announce=True`, writes a single timestamped line announcing the log file path.
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(logs_dir) / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = TimedRotatingFileHandler(str(log_path), when="D", interval=1, backupCount=7, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)

    if announce:
        logger.info("Logging to %s", log_path)
    return logger, str(log_path)


def append_config_block(
    log_path: str,
    config: Mapping[str, Any],
    title: str = "CONFIG",
    sort_keys: bool = True,
    separator_char: str = "=",
) -> None:
    """Append a timestamp-free configuration block to the log file.

    Example output:
    ================================================================
    CONFIG
    epochs              : 100
    lr                  : 0.0001
    ...
    ================================================================
    """
    # Sort keys for stable logs
    items: Iterable[tuple[str, Any]] = config.items()
    if sort_keys:
        items = sorted(items, key=lambda kv: str(kv[0]))

    key_width = max((len(str(k)) for k, _ in items), default=8)
    line = separator_char * 63
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + line + "\n")
        f.write(str(title).upper() + "\n")
        for k, v in items:
            if v is None:
                continue
            f.write(f"{str(k):<{key_width}} : {v}\n")
        f.write(line + "\n\n")
