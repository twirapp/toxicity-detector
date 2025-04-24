import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import device, model_path, threshold
from .logger import log_prediction, log_pytorch_info, logger
from .metrics import *
from .utils import clear_text, measure_time

loop = asyncio.get_running_loop()
loop.set_default_executor(ThreadPoolExecutor())


# Trigger functions
def _predict_toxicity_argmax(logits: Tensor) -> bool:
    return bool(torch.argmax(logits, dim=1).item())


def _predict_toxicity_threshold(logits: Tensor) -> bool:
    return bool(logits[0, 0] < threshold)


predict_func = (
    _predict_toxicity_threshold if threshold != 0 else _predict_toxicity_argmax
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
if device:
    device = torch.device(device)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

log_pytorch_info(device)

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
