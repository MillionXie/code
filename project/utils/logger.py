import logging
from pathlib import Path

from utils.io import ensure_dir


def create_logger(name: str, outdir: str | Path, filename: str = "run.log") -> logging.Logger:
    ensure_dir(outdir)
    log_path = Path(outdir) / filename

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_args(logger: logging.Logger, args: object) -> None:
    logger.info("Arguments:")
    for key, value in sorted(vars(args).items()):
        logger.info("  %s: %s", key, value)
