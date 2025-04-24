import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import cpu_count, model_path, threshold
from .metrics import *
from .utils import clear_text, measure_time

loop = asyncio.get_running_loop()
loop.set_default_executor(ThreadPoolExecutor())

# Initializing logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Trigger functions
def _predict_toxicity_argmax(logits: Tensor) -> bool:
    return bool(torch.argmax(logits, dim=1).item())


def _predict_toxicity_threshold(logits: Tensor) -> bool:
    return bool(logits[0, 0] < threshold)


predict_func = (
    _predict_toxicity_threshold if threshold != 0 else _predict_toxicity_argmax
)


# Logging
def log_prediction(
    text: str, logits: Tensor, result: bool, execution_time: float
) -> None:
    logger.info(
        "%5.3f sec | %6.3f, %6.3f | %-5s | %r",
        execution_time,
        logits[0, 0],
        logits[0, 1],
        str(result),
        text,
    )


# Error handling
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            MODEL_ERRORS.inc()
            logger.error(f"Error calling model: {str(e)}")
            raise

    return wrapper


# Model initialization
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Log PyTorch backends and devices information
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


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
    device, non_blocking=True
)


@measure_time
@error_handler
def call_model(text: str) -> Tensor:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.logits


def sync_predict(text: str) -> bool:
    text = clear_text(text).lower()
    if not text:
        return False

    logits, execution_time = call_model(text)

    MODEL_DURATION.observe(execution_time)
    logit_value = logits[0, 0].item()
    LOGITS_DISTRIBUTION.observe(logit_value)

    result = predict_func(logits)

    log_prediction(text, logits, result, execution_time)
    return result


async def async_predict(text: str) -> bool:
    return await loop.run_in_executor(None, sync_predict, text)
