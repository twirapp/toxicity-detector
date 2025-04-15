import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count, environ

import torch
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Summary
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import clear_text, measure_time

cpu_cores = cpu_count()

# Environment
load_dotenv()

model_path = environ.get("MODEL_PATH", "./model")
threshold = float(environ.get("TOXICITY_THRESHOLD", 0))
metrics_prefix = environ.get("METRICS_PREFIX", "toxicity_detector")
num_threads = int(environ.get("TORCH_THREADS", cpu_cores or 1))

# Configuring Thread Settings
torch.set_num_threads(num_threads)

loop = asyncio.get_running_loop()
loop.set_default_executor(ThreadPoolExecutor())

# Initialize Prometheus metrics
MODEL_ERRORS = Counter(
    f"{metrics_prefix}_model_errors_total", "Total number of model errors"
)

MODEL_DURATION = Summary(
    f"{metrics_prefix}_model_execution_duration_seconds",
    "Model execution duration in seconds",
)

# Define buckets for logits distribution ranging from -10 to 10 with a step of 0.5
# 41 buckets to cover -10 to 10 inclusive
logits_buckets = [round(-10.0 + i * 0.5, 1) for i in range(41)]

LOGITS_DISTRIBUTION = Histogram(
    f"{metrics_prefix}_logits_distribution",
    "Distribution of the first logit value",
    buckets=logits_buckets,
)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log PyTorch backends and devices information
logger.info("CUDA available: %s", torch.cuda.is_available())
logger.info("Current device: %s", device)
logger.info(
    "Number of CPU cores: %s", cpu_cores if cpu_cores is not None else "Unknown"
)
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
