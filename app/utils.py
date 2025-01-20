import logging
import re

logger = logging.getLogger(__name__)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U00002702-\U000027b0"  # Miscellaneous symbols: office supplies, geometric shapes, religious symbols, etc.
    "]+",
    flags=re.UNICODE,
)


def remove_emoji(text: str) -> str:
    try:
        return EMOJI_PATTERN.sub(r"", text)
    except Exception as e:
        logger.warning(
            "Failed to remove emoji from string: %r. Error: %s", text, str(e)
        )
        return text


def clear_text(text: str) -> str:
    if not isinstance(text, str):
        logger.warning(
            "Input is not a string: %r. Type: %s. Returning input unchanged.",
            text,
            type(text),
        )
        return text

    return remove_emoji(text)
