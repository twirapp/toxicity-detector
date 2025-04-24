import logging
from urllib.parse import unquote

import torch
import uvicorn

from .config import cpu_count

__all__ = ["logger", "log_pytorch_info", "log_prediction"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger()


# Log PyTorch backends and devices information
def log_pytorch_info(device: torch.device):
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Current device: %s", device)
    logger.info("Number of CPU cores: %s", cpu_count or "Unknown")

    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)  # type: ignore[attr-defined]
        logger.info("Current CUDA device: %s", torch.cuda.current_device())
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
        logger.info("Number of CUDA devices: %d", torch.cuda.device_count())

    # Check and log additional backends
    logger.info(
        "MPS (Metal Performance Shaders) available: %s",
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    )
    logger.info(
        "CUDNN available: %s",
        hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available(),
    )
    logger.info(
        "MKLDNN available: %s",
        hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available(),
    )
    logger.info(
        "OpenMP available: %s",
        hasattr(torch.backends, "openmp") and torch.backends.openmp.is_available(),
    )


def log_prediction(
    text: str, logits: torch.Tensor, result: bool, execution_time: float
):
    logger.info(
        "%5.3f sec | %6.3f, %6.3f | %-5s | %r",
        execution_time,
        logits[0, 0],
        logits[0, 1],
        str(result),
        text,
    )


class CustomColourizedFormatter(uvicorn.logging.ColourizedFormatter):  # type: ignore[reportAttributeAccessIssue]
    def formatMessage(self, record: logging.LogRecord) -> str:
        levelname = record.levelname

        record.request_line = unquote(record.request_line)  # type: ignore[reportUnknownMemberType]

        if self.use_colors:
            levelname = self.color_level_name(levelname, record.levelno)
            if "color_message" in record.__dict__:
                record.msg = record.__dict__["color_message"]
                record.__dict__["message"] = record.getMessage()
        record.__dict__["levelprefix"] = levelname
        return logging.Formatter.formatMessage(self, record)


class CustomFormatter(uvicorn.logging.AccessFormatter, CustomColourizedFormatter):  # type: ignore[reportAttributeAccessIssue]
    pass


uvicorn_logger = logging.getLogger("uvicorn.access")
console_formatter = CustomFormatter(
    "%(asctime)s | %(levelprefix)s | %(request_line)s | %(status_code)s",
    datefmt="%m-%d %H:%M:%S",
)
uvicorn_logger.handlers[0].setFormatter(console_formatter)
