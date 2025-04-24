from os import cpu_count, environ

from torch import set_num_threads

from .load_dotenv import load_env_to_environ

load_env_to_environ()

cpu_count = cpu_count()

model_path = environ.get("MODEL_PATH", "./model")
threshold = float(environ.get("TOXICITY_THRESHOLD", 0))
metrics_prefix = environ.get("METRICS_PREFIX", "toxicity_detector")
num_threads = int(environ.get("TORCH_THREADS", cpu_count or 1))

set_num_threads(num_threads)
