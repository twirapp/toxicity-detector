import logging
from os import environ

import torch
from dotenv import load_dotenv
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import clear_text

# Initializing logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Environment
load_dotenv()

model_path = environ.get("MODEL_PATH", "./model")
threshold = float(environ.get("TOXICITY_THRESHOLD", 0))


# Trigger functions
def _predict_toxicity_argmax(logits: Tensor) -> bool:
    return bool(torch.argmax(logits, dim=1).item())


def _predict_toxicity_threshold(logits: Tensor) -> bool:
    return bool(logits[0, 0] < threshold)


predict_func = (
    _predict_toxicity_threshold if threshold != 0 else _predict_toxicity_argmax
)


# Logging
def log_prediction(text: str, logits: Tensor, result: bool) -> None:
    logger.info(
        "%6.3f, %6.3f | %5s | %r", logits[0, 0], logits[0, 1], str(result), text
    )


# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)


# Model call
def predict(text: str) -> bool:
    text = clear_text(text).lower()
    if not text:
        return False

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

    result = predict_func(outputs.logits)
    log_prediction(text, outputs.logits, result)
    return result
